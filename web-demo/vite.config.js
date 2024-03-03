import { defineConfig } from 'vite'
import { viteStaticCopy } from 'vite-plugin-static-copy';
import topLevelAwait from "vite-plugin-top-level-await";


export default defineConfig({
  plugins: [
    topLevelAwait(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: '.'
        }
      ]
    }),
  ],
})