import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

const dashboardPath = resolve("public/data/dashboard.json");
const samplePath = resolve("public/data/dashboard.sample.json");

if (!existsSync(dashboardPath)) {
  mkdirSync(dirname(dashboardPath), { recursive: true });
  copyFileSync(samplePath, dashboardPath);
  console.log("Using sample dashboard data for this build.");
} else {
  console.log("Using existing dashboard data for this build.");
}
