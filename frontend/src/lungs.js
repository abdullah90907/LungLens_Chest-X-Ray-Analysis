import * as THREE from 'three';

export function createLungs(canvas, onFallback) {
  let renderer;
  try { renderer = new THREE.WebGLRenderer({canvas, alpha:true, antialias:true, powerPreference:'low-power'}); }
  catch { onFallback(); return {dispose(){}, toggle(){return false;}}; }
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.7));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(37,1,.1,100);
  camera.position.set(0,.15,7.7);
  const anatomy = new THREE.Group(); scene.add(anatomy);
  scene.add(new THREE.AmbientLight(0xffffff,2.1));
  const key = new THREE.DirectionalLight(0xffffff,3); key.position.set(-3,4,5); scene.add(key);
  const fill = new THREE.PointLight(0x28bda7,8,10); fill.position.set(2,0,2); scene.add(fill);

  const shell = new THREE.MeshPhysicalMaterial({color:0x43c4b5,transparent:true,opacity:.19,roughness:.25,metalness:.05,side:THREE.DoubleSide,depthWrite:false});
  const points=[];
  const segments=60, rings=40;
  for (const side of [-1,1]) {
    const vertices=[], indices=[];
    for(let j=0;j<=rings;j++) {
      const t=j/rings, y=-1.4+t*2.7;
      const radius=Math.pow(Math.sin(Math.PI*t),.62);
      for(let i=0;i<=segments;i++) {
        const a=i/segments*Math.PI*2;
        const center=side*(.54+.18*(1-t));
        const x=center+Math.cos(a)*radius*.64*(1.14-.25*t);
        const z=Math.sin(a)*radius*.43;
        vertices.push(x,y,z);
        if(j>1 && j<rings-1 && i%2===0 && j%2===0) points.push(x,y,z);
        if(j<rings && i<segments) {
          const p=j*(segments+1)+i;
          indices.push(p,p+1,p+segments+1,p+1,p+segments+2,p+segments+1);
        }
      }
    }
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3)); geometry.setIndex(indices); geometry.computeVertexNormals();
    anatomy.add(new THREE.Mesh(geometry,shell));
    const edges=new THREE.LineSegments(new THREE.WireframeGeometry(geometry),new THREE.LineBasicMaterial({color:0x41bba9,transparent:true,opacity:.065})); anatomy.add(edges);
  }
  const pg = new THREE.BufferGeometry(); pg.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
  const dots = new THREE.Points(pg,new THREE.PointsMaterial({color:0x239d97,size:.025,transparent:true,opacity:.72,sizeAttenuation:true})); anatomy.add(dots);
  const airwayMat = new THREE.MeshStandardMaterial({color:0xf2fffc,roughness:.36,metalness:.05});
  const branchMat = new THREE.MeshStandardMaterial({color:0x138e82,roughness:.4,transparent:true,opacity:.9});
  function tube(coords,radius,material=airwayMat) {
    const curve=new THREE.CatmullRomCurve3(coords.map(p=>new THREE.Vector3(...p)));
    anatomy.add(new THREE.Mesh(new THREE.TubeGeometry(curve,20,radius,7,false),material));
  }
  tube([[0,1.75,0],[0,1.25,0],[0,.7,0],[0,.36,.08]],.1);
  for(let i=0;i<8;i++) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(.103,.015,5,24),branchMat);
    ring.rotation.x=Math.PI/2; ring.position.set(0,.65+i*.14,0); anatomy.add(ring);
  }
  for(const side of [-1,1]) {
    tube([[0,.53,.04],[side*.22,.31,.1],[side*.49,.1,.04],[side*.6,-.45,0]],.073);
    for(let n=0;n<6;n++) {
      const y=.58-n*.29;
      const start=[side*.52,y-.25,.04];
      const end=[side*(.76+Math.sin(n*1.3)*.19),y-.56,Math.cos(n*2)*.25];
      tube([start,[side*.63,y-.38,.08],end],.027,branchMat);
      for(let b=0;b<2;b++) tube([end,[end[0]+side*.12,end[1]-.13,end[2]+(b?-.13:.13)],[end[0]+side*.17,end[1]-.24,end[2]+(b?-.16:.16)]],.012,branchMat);
    }
  }
  const halo=new THREE.Mesh(new THREE.TorusGeometry(2.05,.006,4,130),new THREE.MeshBasicMaterial({color:0x239d97,transparent:true,opacity:.23}));
  halo.rotation.x=1.12; halo.rotation.y=.1; halo.position.y=-.15; scene.add(halo);
  const halo2=halo.clone(); halo2.rotation.x=.3; halo2.rotation.y=1.15; scene.add(halo2);
  const scattered=[];
  let seed=17; const rand=()=>{seed=(seed*16807)%2147483647;return(seed-1)/2147483646;};
  for(let i=0;i<65;i++) scattered.push((rand()-.5)*5,(rand()-.5)*4,(rand()-.5)*2);
  const dustGeometry=new THREE.BufferGeometry();dustGeometry.setAttribute('position',new THREE.Float32BufferAttribute(scattered,3));
  const dust=new THREE.Points(dustGeometry,new THREE.PointsMaterial({color:0x2589b6,size:.025,transparent:true,opacity:.45}));scene.add(dust);
  const airflow=[];
  const airflowGeometry=new THREE.SphereGeometry(.035,8,8);
  const airflowMaterial=new THREE.MeshBasicMaterial({color:0x21bfaa,transparent:true,opacity:.85});
  for(const side of [-1,1]) {
    const curve=new THREE.CatmullRomCurve3([
      new THREE.Vector3(0,1.72,.13),new THREE.Vector3(0,.65,.14),
      new THREE.Vector3(side*.28,.25,.16),new THREE.Vector3(side*.61,-.28,.13),
      new THREE.Vector3(side*.85,-.85,.16),
    ]);
    for(let i=0;i<5;i++) {
      const marker=new THREE.Mesh(airflowGeometry,airflowMaterial);
      anatomy.add(marker);airflow.push({marker,curve,phase:i/5});
    }
  }
  const reduce=matchMedia('(prefers-reduced-motion: reduce)');
  let moving=!reduce.matches, visible=true, disposed=false, frame=0, rotationX=0, rotationY=0, dragging=false, px=0,py=0;
  let elapsed=0,lastTime=performance.now();
  const resize=new ResizeObserver(()=>{const{width,height}=canvas.getBoundingClientRect();renderer.setSize(width,height,false);camera.aspect=width/Math.max(1,height);camera.updateProjectionMatrix();});resize.observe(canvas);
  const observer=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;});observer.observe(canvas);
  const down=e=>{dragging=true;px=e.clientX;py=e.clientY;canvas.setPointerCapture(e.pointerId);};
  const move=e=>{if(dragging){rotationY+=(e.clientX-px)*.007;rotationX=Math.max(-.3,Math.min(.3,rotationX+(e.clientY-py)*.004));px=e.clientX;py=e.clientY;}};
  const up=()=>{dragging=false;};
  canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',up);
  function draw(now) {
    if(disposed)return;
    frame=requestAnimationFrame(draw);
    const delta=Math.min((now-lastTime)*.001,.05);lastTime=now;
    if(!visible)return;
    if(moving)elapsed+=delta;
    const t=elapsed;
    anatomy.rotation.y=rotationY+Math.sin(t*.23)*.28-.1;
    anatomy.rotation.x=rotationX;
    anatomy.position.y=Math.sin(t*.75)*.07;
    anatomy.scale.setScalar(1+Math.sin(t*.8)*.035);
    dust.rotation.y=t*.025;
    halo.rotation.z=Math.sin(t*.2)*.14;
    halo2.rotation.z=-t*.025;
    for(const flow of airflow)flow.marker.position.copy(flow.curve.getPoint((t*.13+flow.phase)%1));
    renderer.render(scene,camera);
  }
  frame=requestAnimationFrame(draw);
  return {
    toggle(){moving=!moving;return moving;},
    dispose(){disposed=true;cancelAnimationFrame(frame);resize.disconnect();observer.disconnect();canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',up);const geometries=new Set(),materials=new Set();scene.traverse(obj=>{if(obj.geometry)geometries.add(obj.geometry);if(obj.material)materials.add(obj.material);});geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());renderer.dispose();}
  };
}
