import { HttpsProxyAgent } from "https-proxy-agent";

export function getOpenAICommonOptions() {
  const options = {};
  if (process.env.PROXY_URL) {
    options.httpAgent = new HttpsProxyAgent(process.env.PROXY_URL);
  }
  return options;
}
