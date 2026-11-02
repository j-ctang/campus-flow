import test from 'node:test';
import assert from 'node:assert/strict';
import { observationFromMask, importResponse } from '../concentration/import_cached.mjs';
const parser = { decodeMaskToRaster: () => Uint8Array.from([1,0,0,1,1,1,0,1,0]) };
test('half-open bounds and robust bottom-band pixel centers', () => {
  const o = observationFromMask({objectId:'4',bounds:{left:10,top:20,right:13,bottom:23},mask:{width:3,height:3}},parser,852,480);
  assert.deepEqual(o.bbox_xyxy,[10,20,13,23]);
  assert.deepEqual(o.foot_pixel,[11.5,22.5]);
});
test('bottom touching retained but flagged; invalid bounds rejected', () => {
  const r={objectId:'4',bounds:{left:10,top:477,right:13,bottom:480},mask:{width:3,height:3}};
  assert.deepEqual(observationFromMask(r,parser,852,480).quality_flags,['touches_bottom']);
  assert.throws(()=>observationFromMask({...r,bounds:{...r.bounds,right:853}},parser,852,480),/bounds/);
});
const fakeParser = {parseVideoStream:()=>({finalResult:Promise.resolve({outcome:{status:'completed'},diagnostics:[],records:[]})}), recordsOfKind:()=>[]};
const response = text => ({status:'completed',output:[{type:'message',content:[{type:'output_text',text}]}]});
test('explicit empty frames preserved and mapped to source time',async()=>{
 const b=await importResponse(response('<0f>\n<1f>\n'),{batch_id:'b',source_start_frame:1200,frame_count:2},fakeParser);
 assert.equal(b.frames.length,2);assert.equal(b.frames[1].source_frame,1201);assert.equal(b.frames[1].preview_time_s,120.1);assert.deepEqual(b.frames[0].observations,[]);
});
test('missing frames and incomplete responses fail visibly',async()=>{
 await assert.rejects(importResponse(response('<0f>'),{frame_count:2},fakeParser),/frames/);
 await assert.rejects(importResponse({...response('<0f>'),status:'incomplete'},{frame_count:1},fakeParser),/completed/);
});
test('parser diagnostics and wrong declared source dimensions rejected',async()=>{
 const diagnosticParser={...fakeParser,parseVideoStream:()=>({finalResult:Promise.resolve({outcome:{status:'completed'},diagnostics:[{code:'bad'}],records:[]})})};
 await assert.rejects(importResponse(response('<0f>'),{frame_count:1},diagnosticParser),/diagnostics/);
 await assert.rejects(importResponse(response('<0f>0<|box;x1=0;y1=0;x2=2;y2=2;w=10;h=10|>'),{frame_count:1},fakeParser),/dimensions/);
});
test('verified cache identity rejects swapped and unknown responses', async () => {
 const { verifiedCacheIdentity } = await import('../concentration/import_cached.mjs');
 const first='987727ed43aee990cf92c237fcb9cd011b21c0fe4b70b0aef524c5b530963c98';
 const second='d8bb87a4ee8b1ae73499a9ea6e24192f37624ece3219c8da8b1e2c94b3097a5f';
 assert.deepEqual(verifiedCacheIdentity(first,1200),{batch_id:'scout-1200',source_start_frame:1200});
 assert.deepEqual(verifiedCacheIdentity(second,3000),{batch_id:'scout-3000',source_start_frame:3000});
 assert.throws(()=>verifiedCacheIdentity(second,1200),/mismatch/);
 assert.throws(()=>verifiedCacheIdentity(first,3000),/mismatch/);
 assert.throws(()=>verifiedCacheIdentity('0'.repeat(64),1200),/Unknown/);
});
test('response evidence retains model and encoding without inventing prompt',async()=>{
 const b=await importResponse({...response('<0f>'),model:'sam-3.1',metadata:{mask_encoding:'one_bit'},instructions:null},{frame_count:1},fakeParser);
 assert.equal(b.model,'sam-3.1');assert.equal(b.mask_encoding,'one_bit');assert.equal(b.prompt,null);
});
