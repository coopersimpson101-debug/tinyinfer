// Rasterise an SVG to PNG: node render_png.mjs in.svg out.png [width]
// Needs @resvg/resvg-js (npm i @resvg/resvg-js) on the module path.
import { Resvg } from '@resvg/resvg-js';
import { readFileSync, writeFileSync } from 'fs';
const [, , src, dst, w] = process.argv;
const r = new Resvg(readFileSync(src), {
  fitTo: w ? { mode: 'width', value: +w } : { mode: 'original' },
  font: { loadSystemFonts: true, defaultFontFamily: 'Consolas', monospaceFamily: 'Consolas' },
});
writeFileSync(dst, r.render().asPng());
console.log('wrote', dst);
