// jest.config.js
module.exports = {
  preset: 'ts-jest', //
  testEnvironment: 'node',
  testMatch: ['**/src/**/*.test.ts'],
  moduleFileExtensions: ['ts', 'js', 'json', 'node'],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
};
