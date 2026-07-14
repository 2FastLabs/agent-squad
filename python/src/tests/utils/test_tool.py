import pytest
from agent_squad.utils import AgentTools, AgentTool, AgentToolCallbacks, ToolResult, UIPayload
from agent_squad.types import AgentProviderType, ConversationMessage, ParticipantRole
from anthropic import Anthropic
from anthropic.types import ToolUseBlock

def _tool_hanlder(input: str) -> str:
    """
    Prints the input string and returns.
    This is a test tool handler.

    :param input: the input string to return within a sentence.
    :return: the formatted output string.
    """
    return f'This is a {input} tool hanlder'


async def fetch_weather_data(latitude:str, longitude:str):
    """
    Fetches weather data for the given latitude and longitude using the Open-Meteo API.
    Returns the weather data or an error message if the request fails.

    :param latitude: the latitude of the location

    :param longitude: the longitude of the location

    :return: The weather data or an error message.
    """

    return f'Weather data for {latitude}, {longitude}'

def test_tools_without_description():
    tools = AgentTools([AgentTool(
        name="test",
        func=_tool_hanlder
    )])

    for tool in tools.tools:
        assert tool.name == "test"
        assert tool.func_description == """Prints the input string and returns.
This is a test tool handler."""
        assert tool.properties == {'input': {'description': 'the input string to return within a sentence.','type': 'string'}}

def test_tools_with_description():
    tools = AgentTools([AgentTool(
        name="test",
        description="This is a test description.",
        func=_tool_hanlder
    )])

    for tool in tools.tools:
        assert tool.name == "test"
        assert tool.func_description == "This is a test description."
        assert tool.properties == {'input': {'description': 'the input string to return within a sentence.','type': 'string'}}
        assert tool.to_bedrock_format() == {
            'toolSpec': {
                'name': 'test',
                'description': 'This is a test description.',
                'inputSchema': {
                    'json': {
                        'type': 'object',
                        'properties': {
                            'input': {
                                'description': 'the input string to return within a sentence.',
                                'type': 'string'
                            }
                        },
                        'required': ['input']
                    }
                }
            }
        }
        assert tool.to_claude_format() == {
            'name': 'test',
            'description': 'This is a test description.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'input': {
                        'description': 'the input string to return within a sentence.',
                        'type': 'string'
                    }
                },
                'required': ['input']
            }
        }

        assert tool.to_openai_format() == {
            'type': 'function',
            'function': {
                'name': 'test',
                'description': 'This is a test description.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'input': {
                            'description': 'the input string to return within a sentence.',
                            'type': 'string'
                        }
                    },
                    'required': ['input'],
                    'additionalProperties': False
                }
            }
        }


def test_tools_format():
    tools = AgentTools([AgentTool(
        name="weather",
        func=fetch_weather_data
    )])

    for tool in tools.tools:
        assert tool.name == "weather"
        assert tool.func_description == """Fetches weather data for the given latitude and longitude using the Open-Meteo API.
Returns the weather data or an error message if the request fails."""
        assert tool.properties == {'latitude': {'description': 'the latitude of the location', 'type': 'string'},'longitude': {'description': 'the longitude of the location', 'type': 'string'}}
        assert tool.to_bedrock_format() == {
            'toolSpec': {
                'name': 'weather',
                'description': 'Fetches weather data for the given latitude and longitude using the Open-Meteo API.\nReturns the weather data or an error message if the request fails.',
                'inputSchema': {
                    'json': {
                        'type': 'object',
                        'properties': {
                            'latitude': {
                                'description': 'the latitude of the location',
                                'type': 'string'
                            },
                            'longitude': {
                                'description': 'the longitude of the location',
                                'type': 'string'
                            }
                        },
                        'required': ['latitude', 'longitude']
                    }
                }
            }
        }

        assert tool.to_claude_format() == {
            'name': 'weather',
            'description': 'Fetches weather data for the given latitude and longitude using the Open-Meteo API.\nReturns the weather data or an error message if the request fails.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'latitude': {
                        'description': 'the latitude of the location',
                        'type': 'string'
                    },
                    'longitude': {
                        'description': 'the longitude of the location',
                        'type': 'string'
                    }
                },
                'required': ['latitude', 'longitude']
            }
        }

        assert tool.to_openai_format() == {
            'type': 'function',
            'function': {
                'name': 'weather',
                'description': 'Fetches weather data for the given latitude and longitude using the Open-Meteo API.\nReturns the weather data or an error message if the request fails.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'latitude': {
                            'description': 'the latitude of the location',
                            'type': 'string'
                        },
                        'longitude': {
                            'description': 'the longitude of the location',
                            'type': 'string'
                        }
                    },
                    'required': ['latitude', 'longitude'],
                    'additionalProperties': False
                }
            }
        }


@pytest.mark.asyncio
async def test_tool_handler_bedrock():
    tools = AgentTools([AgentTool(
        name="test",
        func=_tool_hanlder
    )])

    tool_message = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{
            'toolUse': {
                'name': 'test',
                'toolUseId': '123',
                'input': {
                    'input': 'hello'
                }
            }
        }])
    response = await tools.tool_handler(AgentProviderType.BEDROCK.value, tool_message, [])
    assert isinstance(response, ConversationMessage) is True
    assert response.role == ParticipantRole.USER.value
    assert response.content[0]['toolResult'] == {'toolUseId': '123', 'content': [{'text': 'This is a hello tool hanlder'}]}

    tools = AgentTools([AgentTool(
        name="weather",
        func=fetch_weather_data
    )])

    tool_message = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{
            'toolUse': {
                'name': 'weather',
                'toolUseId': '456',
                'input': {
                    'latitude': '55.5',
                    'longitude': '37.5'
                }
            }
        }])

    response = await tools.tool_handler(AgentProviderType.BEDROCK.value, tool_message, [])
    assert isinstance(response, ConversationMessage) is True
    assert response.role == ParticipantRole.USER.value
    assert response.content[0]['toolResult'] == {'toolUseId': '456', 'content': [{'text': 'Weather data for 55.5, 37.5'}]}

@pytest.mark.asyncio
async def test_tool_handler_anthropic():
    tools = AgentTools([AgentTool(
        name="test",
        func=_tool_hanlder
    )])

    tool_message = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[ToolUseBlock(name='test',  type='tool_use', id='123', input={'input': 'hello'})])

    response = await tools.tool_handler(AgentProviderType.ANTHROPIC.value, tool_message, [])
    assert response.get('role') == ParticipantRole.USER.value
    assert response.get('content')[0] == {'type':'tool_result', 'tool_use_id': '123', 'content': 'This is a hello tool hanlder'}

    tools = AgentTools([AgentTool(
        name="weather",
        func=fetch_weather_data
    )])

    tool_message = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[ToolUseBlock(name='weather',  type='tool_use', id='456', input={
                    'latitude': '55.5',
                    'longitude': '37.5'
                })])

    response = await tools.tool_handler(AgentProviderType.ANTHROPIC.value, tool_message, [])
    assert response.get('role') == ParticipantRole.USER.value
    assert response.get('content')[0] == {'type':'tool_result', 'tool_use_id': '456', 'content': 'Weather data for 55.5, 37.5'}


def test_tools_format():
    tools = AgentTools([AgentTool(
        name="weather",
        func=fetch_weather_data
    ),
    AgentTool(
        name="test",
        func=_tool_hanlder
    )])

    tools_bedrock_format = tools.to_bedrock_format()
    assert tools_bedrock_format == [
        {
            'toolSpec': {
                'name': 'weather',
                'description': 'Fetches weather data for the given latitude and longitude using the Open-Meteo API.\nReturns the weather data or an error message if the request fails.',
                'inputSchema': {
                    'json': {
                        'type': 'object',
                        'properties': {
                            'latitude': {
                                'description': 'the latitude of the location',
                                'type': 'string'
                            },
                            'longitude': {
                                'description': 'the longitude of the location',
                                'type': 'string'
                            }
                        },
                        'required': ['latitude', 'longitude']
                    }
                }
            }
        },
        {
            'toolSpec': {
                'name': 'test',
                'description': 'Prints the input string and returns.\nThis is a test tool handler.',
                'inputSchema': {
                    'json': {
                        'type': 'object',
                        'properties': {
                            'input': {
                                'description': 'the input string to return within a sentence.',
                                'type': 'string'
                            }
                        },
                        'required': ['input']
                    }
                }
            }
        }
    ]

    tools_claude_format = tools.to_claude_format()
    assert tools_claude_format == [
        {
            'name': 'weather',
            'description': 'Fetches weather data for the given latitude and longitude using the Open-Meteo API.\nReturns the weather data or an error message if the request fails.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'latitude': {
                        'description': 'the latitude of the location',
                        'type': 'string'
                    },
                    'longitude': {
                        'description': 'the longitude of the location',
                        'type': 'string'
                    }
                },
                'required': ['latitude', 'longitude']
            }
        },
        {
            'name': 'test',
            'description': 'Prints the input string and returns.\nThis is a test tool handler.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'input': {
                        'description': 'the input string to return within a sentence.',
                        'type': 'string'
                    }
                },
                'required': ['input']
            }
        }
    ]


def tool_with_enums(latitude:str, longitude:str, units:str):
    """
    Fetches weather data for the given latitude and longitude using the Open-Meteo API.
    Returns the weather data or an error message if the request fails.

    :param latitude: the latitude of the location

    :param longitude: the longitude of the location

    :param units: the units of the weather data

    :return: The weather data or an error message.
    """

    return f'Weather data for {latitude}, {longitude} in {units}'

def test_tool_with_enums():
    tool = AgentTool(
        name="weather_tool",
        func=tool_with_enums,
        enum_values={"units": ["celsius", "fahrenheit"]}
    )

    assert tool.enum_values == {"units": ["celsius", "fahrenheit"]}
    assert tool.to_bedrock_format() == {
        'toolSpec': {
            'name': 'weather_tool',
            'description': 'Fetches weather data for the given latitude and longitude using the Open-Meteo API.\nReturns the weather data or an error message if the request fails.',
            'inputSchema': {
                'json': {
                    'type': 'object',
                    'properties': {
                        'latitude': {
                            'description': 'the latitude of the location',
                            'type': 'string'
                        },
                        'longitude': {
                            'description': 'the longitude of the location',
                            'type': 'string'
                        },
                        'units': {
                            'description': 'the units of the weather data',
                            'enum': ['celsius', 'fahrenheit'],
                            'type': 'string'
                        }
                    },
                    'required': ['latitude', 'longitude', 'units']
                }
            }
        }
    }


def test_tool_with_properties():
    tool = AgentTool(
        name="weather_tool",
        func=tool_with_enums,
        description="Fetches weather data for the given latitude and longitude using the Open-Meteo API.\nReturns the weather data or an error message if the request fails.",
        properties={
            "latitude": {
                "type": "string",
                "description": "the latitude of the location"
            },
            "longitude": {
                "type": "string",
                "description": "the longitude of the location"
            },
            "units": {
                "type": "string",
                "description": "the units of the weather data",
            }
        },
        enum_values={"units": ["celsius", "fahrenheit"]}
    )

    assert tool.enum_values == {"units": ["celsius", "fahrenheit"]}
    assert tool.properties == {
        "latitude": {
            "type": "string",
            "description": "the latitude of the location"
        },
        "longitude": {
            "type": "string",
            "description": "the longitude of the location"
        },
        "units": {
            "type": "string",
            "description": "the units of the weather data",
            "enum": ["celsius", "fahrenheit"]
        }
    }

    assert tool.to_bedrock_format() == {
        'toolSpec': {
            'name': 'weather_tool',
            'description': 'Fetches weather data for the given latitude and longitude using the Open-Meteo API.\nReturns the weather data or an error message if the request fails.',
            'inputSchema': {
                'json': {
                    'type': 'object',
                    'properties': {
                        'latitude': {
                            'description': 'the latitude of the location',
                            'type': 'string'
                        },
                        'longitude': {
                            'description': 'the longitude of the location',
                            'type': 'string'
                        },
                        'units': {
                            'description': 'the units of the weather data',
                            'enum': ['celsius', 'fahrenheit'],
                            'type': 'string'
                        }
                    },
                    'required': ['latitude', 'longitude', 'units']
                }
            }
        }
    }

@pytest.mark.asyncio
async def test_tool_not_found():
    try:
        tools = AgentTools([AgentTool(
            name="weather",
            func=fetch_weather_data
        )])
        await tools._process_tool("test", {'test':'value'})
    except Exception as e:
        assert str(e) == f"Tool weather not found"


def test_get_tool_use_block():
    tools = AgentTools([AgentTool(
        name="weather",
        func=fetch_weather_data
    )])
    response = tools._get_tool_use_block("test", {'test':'value'})
    assert response == None


def test_no_func():
    try:
        tools = AgentTools([AgentTool(
            name="weather",
        )])
    except Exception as e:
        assert str(e) == "Function must be provided"

@pytest.mark.asyncio
async def test_no_tool_block():
    try:
        tools = AgentTools([AgentTool(
            name="weather",
            func=fetch_weather_data
        )])
        message = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=None)
        response = await tools.tool_handler(AgentProviderType.BEDROCK.value, message, [])
    except Exception as e:
        assert str(e) == "No content blocks in response"

@pytest.mark.asyncio
async def test_no_tool_use_block():
    tools = AgentTools([AgentTool(
        name="weather",
        func=fetch_weather_data
    )])
    message = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text'}])
    response = await tools.tool_handler(AgentProviderType.BEDROCK.value, message, [])
    assert isinstance(response, ConversationMessage)
    assert response.role == ParticipantRole.USER.value
    assert response.content == []


def test_self_param():
    def _handler(self, tool_input):
        return tool_input
    tools = AgentTools([AgentTool(
        name="test",
        func=_handler
    )])


@pytest.mark.asyncio
async def test_tool_handler_routes_toolresult_content_to_model():
    """A tool returning a ToolResult: only its text reaches the model; callbacks get the full
    object (structured data + widget), which is what GroundedAgent captures."""
    def order_tool(order_id: str):
        return ToolResult(
            content=f"Order {order_id}: shipped",
            structured_content={"order_id": order_id, "status": "shipped"},
            ui=UIPayload(resource_uri="ui://order", mime_type="text/html;profile=mcp-app"),
        )

    captured = {}

    class _CB(AgentToolCallbacks):
        async def on_tool_end(self, tool_name, payload_input, output, *a, **k):
            captured["output"] = output

    tools = AgentTools([AgentTool(name="get_order", func=order_tool)], callbacks=_CB())
    msg = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{
        "toolUse": {"name": "get_order", "toolUseId": "1", "input": {"order_id": "42"}}}])

    response = await tools.tool_handler(AgentProviderType.BEDROCK.value, msg, [])
    # Only the text content reaches the model — not the structured data or the widget.
    assert response.content[0]["toolResult"]["content"] == [{"text": "Order 42: shipped"}]
    # Callbacks receive the full ToolResult (widget included).
    assert isinstance(captured["output"], ToolResult)
    assert captured["output"].ui.resource_uri == "ui://order"


@pytest.mark.asyncio
async def test_tool_handler_toolresult_empty_content_falls_back_to_structured():
    """A ToolResult with no text: the model gets the JSON of structured_content, not a blank."""
    def t(x: str):
        return ToolResult(content="", structured_content={"x": x})

    tools = AgentTools([AgentTool(name="t", func=t)])
    msg = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{
        "toolUse": {"name": "t", "toolUseId": "1", "input": {"x": "hi"}}}])
    response = await tools.tool_handler(AgentProviderType.BEDROCK.value, msg, [])
    assert response.content[0]["toolResult"]["content"] == [{"text": '{"x": "hi"}'}]


@pytest.mark.asyncio
async def test_tool_handler_toolresult_anthropic():
    """The ToolResult routing applies on the Anthropic provider too."""
    def order_tool(order_id: str):
        return ToolResult(content=f"Order {order_id}", structured_content={"id": order_id})

    tools = AgentTools([AgentTool(name="get_order", func=order_tool)])
    msg = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[
        ToolUseBlock(name="get_order", type="tool_use", id="9", input={"order_id": "7"})])
    response = await tools.tool_handler(AgentProviderType.ANTHROPIC.value, msg, [])
    assert response.get("content")[0] == {"type": "tool_result", "tool_use_id": "9", "content": "Order 7"}




