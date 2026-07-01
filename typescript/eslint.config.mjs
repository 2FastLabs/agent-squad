// Flat config for ESLint v9. Mirrors the former .eslintrc.js:
// eslint:recommended + @typescript-eslint/recommended + prettier, with the same rule overrides.
import js from '@eslint/js';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import prettier from 'eslint-config-prettier';

export default [
  js.configs.recommended,
  // Sets the TS parser, registers the plugin, disables core rules TS supersedes, and applies
  // the recommended TS rules (self-contained array).
  ...tsPlugin.configs['flat/recommended'],
  {
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/ban-ts-comment': 'off',
    },
  },
  // Must come last so it disables any formatting rules enabled above.
  prettier,
];
