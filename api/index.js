// Vercel serverless entrypoint: wrap the Express app as a handler.
const app = require("../index");

module.exports = (req, res) => {
  return app(req, res);
};
