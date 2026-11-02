import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

// Fixed prototype inputs: experiment history associates the first scout cache with
// cam0 frames 1200..1299 and the second with 3000..3099. These hashes bind
// those previously established associations to exact bytes, never filenames.
const VERIFIED_CACHES = new Map([
  ['987727ed43aee990cf92c237fcb9cd011b21c0fe4b70b0aef524c5b530963c98', {batch_id:'scout-1200',source_start_frame:1200}],
  ['d8bb87a4ee8b1ae73499a9ea6e24192f37624ece3219c8da8b1e2c94b3097a5f', {batch_id:'scout-3000',source_start_frame:3000}],
]);

export function verifiedCacheIdentity(digest, expectedStartFrame) {
  const identity=VERIFIED_CACHES.get(digest);
  if(!identity) throw Error('Unknown cached response digest');
  if(identity.source_start_frame!==expectedStartFrame) throw Error('Cached response source-frame mismatch');
  return {...identity};
}

export function observationFromMask(record, parser, width, height) {
  const {left,top,right,bottom}=record.bounds;
  if (![left,top,right,bottom].every(Number.isInteger) || left<0 || top<0 || right>width || bottom>height || right<=left || bottom<=top) throw Error('Invalid mask bounds');
  const w=right-left,h=bottom-top;
  if(record.mask.width!==w || record.mask.height!==h) throw Error('Invalid mask dimensions');
  const raster=parser.decodeMaskToRaster(record.mask);
  if(raster.length!==w*h) throw Error('Invalid decoded mask dimensions');
  let last=-1;
  for(let y=h-1;y>=0 && last<0;y--) for(let x=0;x<w;x++) if(raster[y*w+x]) {last=y;break;}
  if(last<0) throw Error('Empty observation mask');
  const xs=[];
  // Lowest five percent of occupied mask height, at least one pixel row.
  let first=0; while(first<last && !raster.subarray(first*w,(first+1)*w).some(Boolean)) first++;
  const band=Math.max(1,Math.ceil((last-first+1)*0.05));
  for(let y=Math.max(first,last-band+1);y<=last;y++) for(let x=0;x<w;x++) if(raster[y*w+x]) xs.push(left+x+0.5);
  xs.sort((a,b)=>a-b); const mid=Math.floor(xs.length/2);
  const x=xs.length%2?xs[mid]:(xs[mid-1]+xs[mid])/2;
  return {local_track_id:String(record.objectId),bbox_xyxy:[left,top,right,bottom],foot_pixel:[x,top+last+0.5],foot_method:'mask_bottom_5pct_median_x_pixel_center',quality_flags:bottom===height?['touches_bottom']:[]};
}

export async function importResponse(response, options, parser) {
  const {batch_id,source_start_frame,width=852,height=480,fps=10,frame_count=100}=options;
  if(response.status!=='completed' || response.error) throw Error('Cached response must be completed');
  const contents=(response.output??[]).flatMap(item=>item.content??[]);
  if(contents.some(c=>c.type==='refusal')) throw Error('Cached response contains refusal');
  const text=contents.filter(c=>c.type==='output_text').map(c=>c.text).join('\n');
  const markers=[...text.matchAll(/<(\d+)f>/g)].map(m=>Number(m[1]));
  if(markers.length!==frame_count || markers.some((n,i)=>n!==i)) throw Error('Expected complete, ordered clip frames');
  const dims=[...text.matchAll(/<\|box;([^|]+)\|>/g)];
  for(const [,fields] of dims) {
    const values=Object.fromEntries(fields.split(';').map(f=>f.split('=')));
    if(Number(values.w)!==width || Number(values.h)!==height) throw Error('Invalid source dimensions');
  }
  async function* events(){yield {type:'response.output_text.done',item_id:'cached',output_index:0,content_index:0,text};yield {type:'response.completed'};}
  const result=await parser.parseVideoStream(events()).finalResult;
  if(result.outcome.status!=='completed' || result.diagnostics.length) throw Error('Parser incomplete or diagnostics present');
  const frames=Array.from({length:frame_count},(_,i)=>({clip_frame:i,source_frame:source_start_frame+i,preview_time_s:(source_start_frame+i)/fps,observations:[]}));
  for(const r of parser.recordsOfKind(result.records,'mask')) {
    const frame=parser.frameIndexOf(r);
    if(!Number.isInteger(frame)||frame<0||frame>=frame_count) throw Error('Invalid mask frame');
    if(frames[frame].observations.some(o=>o.local_track_id===String(r.objectId))) throw Error('Duplicate track in frame');
    frames[frame].observations.push(observationFromMask(r,parser,width,height));
  }
  const boxes=parser.recordsOfKind(result.records,'box');
  if(boxes.length!==frames.reduce((n,f)=>n+f.observations.length,0)) throw Error('Box missing corresponding mask');
  // These response objects expose model and encoding, but omit request prompts.
  const model=typeof response.model==='string'?response.model:null;
  const mask_encoding=typeof response.metadata?.mask_encoding==='string'?response.metadata.mask_encoding:null;
  const prompt=typeof response.prompt==='string'?response.prompt:null;
  return {batch_id,source_start_frame,width,height,fps,model,mask_encoding,prompt,frames};
}

export async function main(args=process.argv.slice(2)) {
  if(args.length!==3) throw Error('Usage: node import_cached.mjs RESPONSE_A RESPONSE_B OUTPUT');
  const parserPath=process.env.SAM_PARSER_PATH;
  const parser=await import(parserPath?pathToFileURL(path.resolve(parserPath,'dist/index.js')).href:'@meta-sam/parser');
  const packagePath=parserPath?path.join(parserPath,'package.json'):new URL('../package.json',import.meta.resolve('@meta-sam/parser'));
  const version=JSON.parse(await fs.readFile(packagePath,'utf8')).version;
  if(version!=='0.1.0') throw Error('Expected parser version 0.1.0');
  const batches=[];
  for(let i=0;i<2;i++) {
    const bytes=await fs.readFile(args[i]);
    const digest=createHash('sha256').update(bytes).digest('hex');
    const identity=verifiedCacheIdentity(digest,i===0?1200:3000);
    const batch=await importResponse(JSON.parse(bytes),identity,parser);
    batch.response_sha256=digest;batches.push(batch);
  }
  await fs.mkdir(path.dirname(args[2]),{recursive:true});
  await fs.writeFile(args[2],JSON.stringify({schema_version:1,parser_version:version,batches},null,2)+'\n');
}
if(process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href===import.meta.url) main().catch(error=>{console.error(error.message);process.exitCode=1;});
