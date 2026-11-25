import * as cdk from 'aws-cdk-lib';
import * as nodejs from 'aws-cdk-lib/aws-lambda-nodejs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import { Construct } from 'constructs';
import * as path from "path";
import { LexAgentConstruct } from './lex-agent-construct';
import { BedrockKnowledgeBase } from './knowledge-base-construct';
import {BedrockKnowledgeBaseModels } from './constants';


export class ChatDemoStack extends cdk.Stack {
  public multiAgentLambdaFunctionUrl: cdk.aws_lambda.FunctionUrl;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const enableLexAgent = this.node.tryGetContext('enableLexAgent');

    let lexAgent = null;
    let lexAgentConfig = {};
    if (enableLexAgent === true){
      lexAgent = new LexAgentConstruct(this, "LexAgent");
      lexAgentConfig = {
        botId: lexAgent.lexBotId,
        botAliasId: lexAgent.lexBotAliasId,
        localeId: "en_US",
        description: lexAgent.lexBotDescription,
        name: lexAgent.lexBotName,
      }
    }

    const documentsBucket = new s3.Bucket(this, 'DocumentsBucket', {
      enforceSSL:true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    const assetsPath = path.join(__dirname, "../../../docs/src/content/docs/");
    const assetDoc = s3deploy.Source.asset(assetsPath);

    const assetsTsPath = path.join(__dirname, "../../../typescript/src/");
    const assetTsDoc = s3deploy.Source.asset(assetsTsPath);

    const assetsPyPath = path.join(__dirname, "../../../python/src/agent_squad/");
    const assetPyDoc = s3deploy.Source.asset(assetsPyPath);


    const knowledgeBase = new BedrockKnowledgeBase(this, 'MutiAgentOrchestratorDocKb', {
      kbName:'agent-squad-doc-kb',
      assetFiles:[],
      embeddingModel: BedrockKnowledgeBaseModels.TITAN_EMBED_TEXT_V1,
    });

    const maoFilesDeployment = new s3deploy.BucketDeployment(this, "DeployDocumentation", {
      sources: [assetDoc, assetTsDoc, assetPyDoc],
      destinationBucket: documentsBucket,
    });

    knowledgeBase.addS3Permissions(documentsBucket.bucketName);
    knowledgeBase.createAndSyncDataSource(documentsBucket.bucketArn);

    const powerToolsTypeScriptLayer = lambda.LayerVersion.fromLayerVersionArn(
      this,
      "powertools-layer-ts",
      `arn:aws:lambda:${
        cdk.Stack.of(this).region
      }:094274105915:layer:AWSLambdaPowertoolsTypeScriptV2:2`
    );

    const basicLambdaRole = new iam.Role(this, "BasicLambdaRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
    });

    basicLambdaRole.addManagedPolicy(
      iam.ManagedPolicy.fromManagedPolicyArn(
        this,
        "basicLambdaRoleAWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
      )
    );
    const sessionTable = new dynamodb.Table(this, "SessionTable", {
      partitionKey: { name: "PK", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      sortKey: { name: "SK", type: dynamodb.AttributeType.STRING },
      timeToLiveAttribute: "TTL",
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const pythonLambda = new lambda.Function(this, "PythonLambda", {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "lambda.lambda_handler",
      code: lambda.Code.fromAsset(
        path.join(__dirname, "../lambda/find-my-name")
      ),
      memorySize: 128,
      timeout: cdk.Duration.seconds(10),
    })

    const multiAgentLambdaFunction = new nodejs.NodejsFunction(
      this,
      "MultiAgentLambda",
      {
        entry: path.join(
          __dirname,
          "../lambda/multi-agent/index.ts"
        ),
        runtime: lambda.Runtime.NODEJS_20_X,
        role: basicLambdaRole,
        memorySize: 2048,
        timeout: cdk.Duration.minutes(5),
        layers: [powerToolsTypeScriptLayer],
        environment: {
          POWERTOOLS_SERVICE_NAME: "multi-agent",
          POWERTOOLS_LOG_LEVEL: "DEBUG",
          HISTORY_TABLE_NAME: sessionTable.tableName,
          HISTORY_TABLE_TTL_KEY_NAME: 'TTL',
          HISTORY_TABLE_TTL_DURATION: '3600',
          LEX_AGENT_ENABLED: enableLexAgent.toString(),
          LEX_AGENT_CONFIG: JSON.stringify(lexAgentConfig),
          KNOWLEDGE_BASE_ID: knowledgeBase.knowledgeBase.attrKnowledgeBaseId,
          LAMBDA_AGENTS: JSON.stringify(
            [{description:"This is an Agent to use when you forgot about your own name",name:'Find my name',functionName:pythonLambda.functionName, region:cdk.Aws.REGION}]),
        },
        bundling: {
          minify: false,
          externalModules: [
            //"aws-lambda",
            "@aws-lambda-powertools/logger",
            "@aws-lambda-powertools/parameters",
            //"@aws-sdk/client-ssm",
        ],
        },
      }
    );

    sessionTable.grantReadWriteData(multiAgentLambdaFunction);
    pythonLambda.grantInvoke(multiAgentLambdaFunction);

    multiAgentLambdaFunction.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ],
        resources: [
          `arn:aws:bedrock:${cdk.Aws.REGION}::foundation-model/*`,
          `arn:${cdk.Aws.PARTITION}:bedrock:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0`,
          `arn:${cdk.Aws.PARTITION}:bedrock:::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0`,
        ],
      })
    );

    multiAgentLambdaFunction.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        sid: 'AmazonBedrockKbPermission',
        actions: [
          "bedrock:Retrieve",
          "bedrock:RetrieveAndGenerate"
        ],
        resources: [
          `arn:aws:bedrock:${cdk.Aws.REGION}::foundation-model/*`,
          `arn:${cdk.Aws.PARTITION}:bedrock:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:knowledge-base/${knowledgeBase.knowledgeBase.attrKnowledgeBaseId}`
        ]
      })
    );



    const multiAgentLambdaFunctionUrl = multiAgentLambdaFunction.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
      invokeMode: lambda.InvokeMode.RESPONSE_STREAM,
    });

    this.multiAgentLambdaFunctionUrl = multiAgentLambdaFunctionUrl;

    if (enableLexAgent){
      multiAgentLambdaFunction.addToRolePolicy(
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          sid: 'LexPermission',
          actions: [
            "lex:RecognizeText",
          ],
          resources: [
            `arn:aws:bedrock:${cdk.Aws.REGION}::foundation-model/*`,
            `arn:aws:lex:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:bot-alias/${lexAgent!.lexBotId}/${lexAgent!.lexBotAliasId}`
          ],
        })
      );
    }
  }
}
