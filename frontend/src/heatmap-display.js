export async function decodeAttribution(source) {
  const image=new Image();image.src=source;await image.decode();
  if(image.width>1024||image.height>1024)throw new Error('Heatmap exceeds preview size.');
  const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
  const context=canvas.getContext('2d');context.drawImage(image,0,0);
  const pixels=context.getImageData(0,0,canvas.width,canvas.height).data;
  const values=new Float32Array(canvas.width*canvas.height);
  // The backend encodes normalized attribution as alpha = intensity * 175.
  for(let i=0;i<values.length;i++)values[i]=Math.min(1,pixels[i*4+3]/175);
  return {width:canvas.width,height:canvas.height,values};
}

export function displayAttribution(map, cutoff=0) {
  const canvas=document.createElement('canvas');canvas.width=map.width;canvas.height=map.height;
  const context=canvas.getContext('2d');const image=context.createImageData(map.width,map.height);
  const stops=[[49,115,214],[246,218,99],[244,112,39]];
  for(let i=0;i<map.values.length;i++){
    const value=map.values[i];if(value<=0||value<cutoff)continue;
    const position=value*2,lower=Math.min(Math.floor(position),1),mix=position-lower;
    for(let c=0;c<3;c++)image.data[i*4+c]=Math.round(stops[lower][c]*(1-mix)+stops[lower+1][c]*mix);
    image.data[i*4+3]=Math.round(Math.pow(value,1.05)*220);
  }
  context.putImageData(image,0,0);return canvas.toDataURL('image/png');
}
