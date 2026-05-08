import { defineConfig } from "tsup"

export default defineConfig({
  entry: { index: "src/index.ts" },
  format: ["esm"],
  dts: true,
  clean: true,
  sourcemap: true,
  splitting: false,
  treeshake: true,
  target: "es2022",
  platform: "browser",
  external: ["tiktoken", "xstate", "zod"],
  esbuildOptions(options) {
    options.conditions = ["browser", "import", "module"]
  },
})
