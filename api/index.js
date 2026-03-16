// Vercel serverless entrypoint: reuse the Express app without starting a local listener.
const app = require("../index");
module.exports = app;
