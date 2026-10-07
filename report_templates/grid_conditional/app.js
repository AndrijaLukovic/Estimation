"use strict";
(() => {
  const DATA = JSON.parse(document.getElementById("report-data").textContent);
  const $ = id => document.getElementById(id);
  const COLORS = ["#718096", "#d49436", "#8b5cc7", "#268b89", "#bf6278", "#3676a9"];
  const RP = ["SQ", "PA", "LE", "FE"];
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const fmt = (value, digits=4) => !finite(value) ? "—" :
    (value !== 0 && (Math.abs(value) < 0.0001 || Math.abs(value) >= 1e7) ? value.toExponential(4) : value.toLocaleString("zh-CN", {maximumFractionDigits:digits}));
  const pct = value => finite(value) ? (value*100).toFixed(2)+"%" : "—";
  const badge = (text, style="") => `<span class="badge ${style}">${esc(text)}</span>`;
  const prob = file => file.probability_pass === true ? badge("通过") : file.probability_pass === false ? badge("不通过","bad") : badge("未完全核对","muted");
  const conv = file => file.converged ? badge("LL 容差") : badge(file.exit_reason || "未知","warn");
  const table = (headers, rows) => `<table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(cell=>`<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  function card(label, value, note="") { return `<div class="card"><div class="label">${esc(label)}</div><div class="value num">${esc(value)}</div><div class="muted">${esc(note)}</div></div>`; }
  function download(name, text, type="text/plain;charset=utf-8") {
    const a=document.createElement("a"), url=URL.createObjectURL(new Blob([text],{type}));
    a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function csv(headers, rows) {
    const quote=v=>`"${String(v??"").replace(/"/g,'""')}"`;
    return "\ufeff"+[headers,...rows].map(row=>row.map(quote).join(",")).join("\r\n");
  }
  function svgElement(tag,attrs={},text) {
    const element=document.createElementNS("http://www.w3.org/2000/svg",tag);
    for (const [key,value] of Object.entries(attrs)) element.setAttribute(key,value);
    if (text!==undefined)element.textContent=text;return element;
  }
  function emptyChart(id, title, text="当前筛选没有可绘制的数据") { $(id).innerHTML=`<div class="chart"><h3>${esc(title)}</h3><div class="chart-empty">${esc(text)}</div></div>`; }
  function tooltip(event,text) {
    const tip=$("tooltip");if(!tip)return;tip.textContent=text;tip.style.display="block";
    tip.style.left=Math.min(event.clientX+12,window.innerWidth-320)+"px";tip.style.top=Math.max(5,event.clientY-35)+"px";
  }
  function lineChart(id,title,series,options={}) {
    series=series.map(s=>({...s,points:s.points.filter(p=>finite(p[0])&&finite(p[1]))})).filter(s=>s.points.length);
    if(!series.length)return emptyChart(id,title);
    const host=$(id);host.innerHTML=`<div class="chart"><h3>${esc(title)}</h3></div>`;const wrap=host.firstChild;
    const W=680,H=310,M={left:65,right:24,top:14,bottom:52};
    const points=series.flatMap(s=>s.points);let xmin=options.xmin??Math.min(...points.map(p=>p[0])),xmax=options.xmax??Math.max(...points.map(p=>p[0]));
    let ymin=options.ymin??Math.min(...points.map(p=>p[1])),ymax=options.ymax??Math.max(...points.map(p=>p[1]));
    if(xmin===xmax){xmin-=.5;xmax+=.5;}if(ymin===ymax){const pad=Math.max(1e-6,Math.abs(ymin)*.01);ymin-=pad;ymax+=pad;}
    if(options.ymin===undefined&&options.ymax===undefined){const pad=(ymax-ymin)*.07;ymin-=pad;ymax+=pad;}
    const X=x=>M.left+(x-xmin)/(xmax-xmin)*(W-M.left-M.right),Y=y=>H-M.bottom-(y-ymin)/(ymax-ymin)*(H-M.top-M.bottom);
    const svg=svgElement("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":title});wrap.appendChild(svg);
    for(let i=0;i<=4;i++) {
      const y=ymin+(ymax-ymin)*i/4;svg.appendChild(svgElement("line",{x1:M.left,y1:Y(y),x2:W-M.right,y2:Y(y),stroke:"#e8eef3"}));
      svg.appendChild(svgElement("text",{x:M.left-8,y:Y(y)+4,"text-anchor":"end"},options.yformat?options.yformat(y):fmt(y,3)));
      const x=xmin+(xmax-xmin)*i/4;svg.appendChild(svgElement("text",{x:X(x),y:H-M.bottom+20,"text-anchor":"middle"},options.xformat?options.xformat(x):fmt(x,2)));
    }
    svg.appendChild(svgElement("text",{x:(W+M.left)/2,y:H-5,"text-anchor":"middle"},options.xlabel||""));
    if(options.zero&&ymin<0&&ymax>0)svg.appendChild(svgElement("line",{x1:M.left,y1:Y(0),x2:W-M.right,y2:Y(0),stroke:"#8597a5","stroke-dasharray":"4 4"}));
    series.forEach((s,i)=>{
      const color=s.color||COLORS[i%COLORS.length];svg.appendChild(svgElement("path",{d:s.points.map((p,j)=>`${j?"L":"M"}${X(p[0])},${Y(p[1])}`).join(" "),fill:"none",stroke:color,"stroke-width":s.dashed?1.3:2,"stroke-dasharray":s.dashed?"5 4":""}));
      if(s.points.length<12)s.points.forEach(p=>svg.appendChild(svgElement("circle",{cx:X(p[0]),cy:Y(p[1]),r:3,fill:color})));
    });
    const legend=document.createElement("div");legend.className="legend";
    legend.innerHTML=series.map((s,i)=>`<span><i class="dot" style="background:${s.color||COLORS[i%COLORS.length]}"></i>${esc(s.name)}</span>`).join("");wrap.appendChild(legend);
    svg.addEventListener("pointermove",event=>{
      const bounds=svg.getBoundingClientRect(),px=(event.clientX-bounds.left)/bounds.width*W,x=xmin+(px-M.left)/(W-M.left-M.right)*(xmax-xmin);
      const display=series.filter(s=>!s.dashed).map(s=>{const p=s.points.reduce((a,b)=>Math.abs(a[0]-x)<Math.abs(b[0]-x)?a:b);return `${s.name}: ${fmt(p[1],6)}`;});
      tooltip(event,`${options.xformat?options.xformat(x):fmt(x,3)}\n${display.join("\n")}`);
    });svg.addEventListener("pointerleave",()=>{if($("tooltip"))$("tooltip").style.display="none";});
  }
  function barChart(id,title,rows,options={}) {
    if(!rows.length)return emptyChart(id,title);
    const host=$(id);host.innerHTML=`<div class="chart"><h3>${esc(title)}</h3></div>`;
    const W=680,left=options.left||240,right=80,rowHeight=31,H=rows.length*rowHeight+45;
    const max=Math.max(...rows.map(r=>Math.abs(r.value)),1e-9);const width=W-left-right;
    const svg=svgElement("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":title});host.firstChild.appendChild(svg);
    rows.forEach((row,i)=>{
      const y=i*rowHeight+8,label=row.label.length>38?row.label.slice(0,35)+"…":row.label;
      svg.appendChild(svgElement("text",{x:left-8,y:y+15,"text-anchor":"end"},label));
      const rect=svgElement("rect",{x:left,y,width:Math.max(1,Math.abs(row.value)/max*width),height:20,rx:3,fill:row.color||COLORS[i%COLORS.length]});
      rect.appendChild(svgElement("title",{},`${row.label}: ${fmt(row.value,6)}`));if(row.click){rect.style.cursor="pointer";rect.addEventListener("click",row.click);}svg.appendChild(rect);
      svg.appendChild(svgElement("text",{x:left+Math.abs(row.value)/max*width+5,y:y+15},fmt(row.value,3)));
    });svg.appendChild(svgElement("text",{x:left,y:H-5},options.xlabel||""));
  }
  function histogram(id,title,values,bins=24) {
    values=values.filter(finite);if(!values.length)return emptyChart(id,title);
    let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){lo-=.01;hi+=.01;}
    const counts=Array(bins).fill(0);values.forEach(v=>counts[Math.min(bins-1,Math.floor((v-lo)/(hi-lo)*bins))]++);
    const W=680,H=290,L=45,B=38,top=15,max=Math.max(...counts);
    $(id).innerHTML=`<div class="chart"><h3>${esc(title)}</h3></div>`;const svg=svgElement("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":title});$(id).firstChild.appendChild(svg);
    counts.forEach((n,i)=>{const x=L+i*(W-L-20)/bins,w=(W-L-20)/bins-1,y=H-B-(H-B-top)*n/max;
      const rect=svgElement("rect",{x,y,width:w,height:H-B-y,fill:"#268b89"});rect.appendChild(svgElement("title",{},`${fmt(lo+(hi-lo)*i/bins)}—${fmt(lo+(hi-lo)*(i+1)/bins)}: ${n} 人`));svg.appendChild(rect);});
    svg.appendChild(svgElement("text",{x:L,y:H-12},fmt(lo)));svg.appendChild(svgElement("text",{x:W-20,y:H-12,"text-anchor":"end"},fmt(hi)));
    svg.appendChild(svgElement("text",{x:L,y:12},`最高柱：${max} 人；N=${values.length}`));
  }
  function weighting(p,c) {
    if(p<=0)return 0;if(p>=1)return 1;const v=c.values;
    return c.method==="tk"?Math.pow(p,v.gamma)/Math.pow(Math.pow(p,v.gamma)+Math.pow(1-p,v.gamma),1/v.gamma):Math.exp(-(v.beta??1)*Math.pow(-Math.log(p),v.palpha));
  }
  function ratioSeries(clusters,prefix="") {
    return clusters.map((c,i)=>({name:prefix+c.label,color:COLORS[i%COLORS.length],points:Array.from({length:121},(_,j)=>{
      const x=-2+j*.05;return [x,Math.log10(c.values.lambda)+(c.values.alpha_loss-c.values.alpha_gain)*x];})}));
  }
  function dashboard() {
    const files=DATA.files,selected=new Set();let filtered=[],page=0,sortKey="bic",sortDirection=1;
    $("snapshot").textContent=`生成时间 ${DATA.generated_at} · 数据快照 ${DATA.input_dir}`;
    $("counts").innerHTML=card("全部 pickle",DATA.counts.total)+card("模型结果",DATA.counts.models,`其中等价标签 ${files.filter(f=>f.kind==="model"&&f.equivalent_to).length}`)+card("完成起点",DATA.counts.restarts)+card("转换失败",DATA.counts.errors)+["test1","test2","test3"].map((t,i)=>card(t.toUpperCase(),`${DATA.counts.tests[t]||0} / ${[64,64,16][i]}`,"已保存的模型标签")).join("");
    const integrity=files.filter(f=>f.audit_status==="verified"),maxgap=Math.max(0,...integrity.map(f=>Math.abs(f.ll_gap||0)));
    $("integrity").textContent=`转换 ${DATA.counts.total-DATA.counts.errors}/${DATA.counts.total} 个文件；当前数据复算 ${integrity.length} 个，最大 |LL 复算差|=${fmt(maxgap,8)}。${DATA.source_matches?"估计源码／数据与目录协议一致。":"源码／数据与目录协议未确认一致，未使用不匹配的代码强行复算。"} 边界警示阈值 ${DATA.boundary_tolerance}；所有原始数组与日志保留在各详情页。`;
    if(DATA.counts.errors||!DATA.source_matches)$("integrity").classList.add("warn");
    $("protocol").textContent=JSON.stringify(DATA.protocol,null,2);
    $("progress-only").innerHTML=DATA.progress_only.length?table(["文件","记录状态","迭代","记录 LL"],DATA.progress_only.map(p=>[esc(p.path),esc(p.status),fmt(p.iteration,0),fmt(p.log_likelihood)])):"没有仅进度记录。";
    function passes(file) {
      const s=file.spec||{},q=$("quality").value,kind=$("kind").value;
      if(kind!=="all"&&file.kind!==kind)return false;
      if($("hide-alias").checked&&file.equivalent_to)return false;
      for(const [id,key] of [["test","test"],["pw","pw"],["discount","discounting"],["curvature","curvature"],["memory","memory"],["clusters","C"]])if($(id).value&&String(s[key])!==$(id).value)return false;
      if(q==="prob"&&file.probability_pass!==true)return false;
      if(q==="strict"&&(file.probability_pass!==true||file.near_zero_prelec))return false;
      if(q==="converged"&&!file.converged)return false;
      if(q==="warnings"&&!file.warnings.length)return false;
      if(q==="noaudit"&&file.audit_status==="verified")return false;
      const text=$("search").value.trim().toLowerCase();return !text||`${file.config_id} ${file.restart_id||""} ${file.path}`.toLowerCase().includes(text);
    }
    const sortValue=file=>sortKey==="name"?file.config_id+(file.restart_id||""):file[sortKey];
    function render() {
      filtered=files.filter(passes).sort((a,b)=>{const av=sortValue(a),bv=sortValue(b);if(av==null)return 1;if(bv==null)return -1;return sortDirection*(typeof av==="string"?av.localeCompare(bv):av-bv);});
      const pages=Math.max(1,Math.ceil(filtered.length/40));page=Math.min(page,pages-1);const visible=filtered.slice(page*40,(page+1)*40);
      const heads=[["选",""],["文件／模型","name"],["BIC","bic"],["ΔBIC","delta_bic"],["LL","ll"],["同配置 ΔLL","delta_ll"],["迭代","n_iter"],["收敛",""],["概率",""],["边界数","boundary_count"],["优化器非成功","optimizer_unsuccessful"],["LL 复算差","ll_gap"]];
      $("file-table").innerHTML=`<thead><tr>${heads.map(([label,key])=>`<th ${key?`class="sort" data-sort="${key}"`:""}>${esc(label)}${sortKey===key?(sortDirection===1?" ↑":" ↓"):""}</th>`).join("")}</tr></thead><tbody>${visible.map(file=>`<tr class="${selected.has(file.id)?"selected":""}"><td><input aria-label="选择比较 ${esc(file.config_id)}" type="checkbox" data-select="${file.id}" ${selected.has(file.id)?"checked":""} ${file.kind==="error"?"disabled":""}></td><td class="wrap"><a href="${file.href}" data-open="${file.id}">${esc(file.config_id)}</a><div class="muted">${esc(file.kind==="restart"?file.restart_id:file.kind==="model"?"模型结果":"其他文件")}${file.equivalent_to?" · 等价标签":""}</div></td><td class="num">${fmt(file.bic,2)}</td><td>${fmt(file.delta_bic,2)}</td><td>${fmt(file.ll,4)}</td><td>${fmt(file.delta_ll,4)}</td><td>${fmt(file.n_iter,0)}</td><td>${conv(file)}</td><td>${prob(file)}${file.near_zero_prelec?" "+badge("近零","warn"):""}</td><td>${fmt(file.boundary_count,0)}</td><td>${fmt(file.optimizer_unsuccessful,0)}</td><td>${fmt(file.ll_gap,8)}</td></tr>`).join("")}</tbody>`;
      $("table-status").textContent=`${filtered.length} 个匹配文件 · 已选 ${selected.size} 个`;
      $("page-status").textContent=`${page+1} / ${pages}`;$("prev").disabled=page===0;$("next").disabled=page===pages-1;
      $("file-table").querySelectorAll("[data-sort]").forEach(el=>el.onclick=()=>{if(sortKey===el.dataset.sort)sortDirection*=-1;else{sortKey=el.dataset.sort;sortDirection=sortKey==="ll"?-1:1;}render();});
      $("file-table").querySelectorAll("[data-open]").forEach(el=>el.onclick=event=>{if(event.ctrlKey||event.metaKey)return;event.preventDefault();openFile(files.find(f=>f.id===el.dataset.open));});
      $("file-table").querySelectorAll("[data-select]").forEach(el=>el.onchange=()=>{if(el.checked&&selected.size>=4){el.checked=false;alert("最多选择 4 个文件进行比较。");return;}el.checked?selected.add(el.dataset.select):selected.delete(el.dataset.select);render();renderComparison();});
      renderCharts();
    }
    function openFile(file) {
      $("reader-section").hidden=false;$("reader-title").textContent=file.restart_id?`${file.config_id} · ${file.restart_id}`:file.config_id;
      $("reader").src=file.href;$("open-detail").href=file.href;$("reader-section").scrollIntoView({behavior:"smooth",block:"start"});
    }
    function renderCharts() {
      const models=filtered.filter(f=>f.kind==="model"&&finite(f.delta_bic)),chosen=[];
      for(const test of ["test1","test2","test3"])chosen.push(...models.filter(f=>f.spec.test===test).sort((a,b)=>a.delta_bic-b.delta_bic).slice(0,4));
      barChart("ranking-chart","模型 ΔBIC（各测试基准独立）",chosen.map(f=>({label:f.config_id.replace("commem_","").replace("commonalpha","共同").replace("sepalpha","分开"),value:f.delta_bic,click:()=>openFile(f)})),{left:280,xlabel:"每个测试的前 4 个匹配模型，点击条形查看详情"});
      const configs=new Set(filtered.map(f=>f.config_id)),groups=new Map();
      files.filter(f=>f.kind==="restart"&&configs.has(f.config_id)&&finite(f.delta_ll)).forEach(f=>{if(!groups.has(f.config_id))groups.set(f.config_id,[]);groups.get(f.config_id).push(f);});
      const spreads=[...groups].map(([id,fs])=>({label:id,value:Math.max(...fs.map(f=>f.delta_ll)),fs})).sort((a,b)=>b.value-a.value).slice(0,12);
      barChart("restart-chart","同配置起点的最大 LL 差距",spreads.map(g=>({label:g.label,value:g.value,color:"#d49436",click:()=>{$("kind").value="restart";$("search").value=g.label;page=0;render();}})),{left:280,xlabel:"点击条形筛选该配置的全部起点"});
    }
    function renderComparison() {
      const fs=files.filter(f=>selected.has(f.id));$("comparison-section").hidden=fs.length===0;if(!fs.length)return;
      const groups=new Set(fs.map(f=>f.comparison_group));$("compare-warning").textContent=groups.size>1?"所选文件的 ΔBIC 基准不同。原始 BIC 只有在样本和估值协议一致时才可比较；测试设计也需要分别解释。Test 1 的自由类别编号不保证跨模型对应同一类。":"所选文件属于相同比较组。高后验、低 BIC 与达到 LL 容差均不能证明真实机制或全局最优。";
      const measures=[["BIC",f=>fmt(f.bic,2)],["ΔBIC",f=>fmt(f.delta_bic,2)],["LL",f=>fmt(f.ll,4)],["同配置 ΔLL",f=>fmt(f.delta_ll,4)],["自由参数 k",f=>fmt(f.k,0)],["类别数",f=>fmt(f.spec.C,0)],["收敛",conv],["概率检查",prob],["近零 Prelec",f=>f.near_zero_prelec?"是":"否"],["边界数",f=>fmt(f.boundary_count,0)],["优化器非成功",f=>fmt(f.optimizer_unsuccessful,0)],["归一化 RMSE",f=>fmt(f.normalized_rmse,6)],["ξ 中位数",f=>fmt(f.ksi_median,6)],["后验中位数",f=>fmt(f.posterior_median,6)]];
      $("compare-table").innerHTML=table(["指标",...fs.map((f,i)=>`${i+1} · ${f.config_id} ${f.kind==="restart"?f.restart_id:""}`)],measures.map(([label,value])=>[esc(label),...fs.map(value)]));
      const curves=fs.flatMap((f,i)=>ratioSeries(f.clusters,`${i+1} · `));lineChart("compare-chart","各文件、各类别的 log10 A(d)：0 表示等额总幅度相同",curves,{xlabel:"金额 d（英镑，对数刻度）",xformat:x=>`£${fmt(10**x,2)}`,zero:true});
    }
    for(const id of ["kind","test","pw","discount","curvature","memory","clusters","quality","hide-alias"])$(id).onchange=()=>{page=0;render();};
    $("search").oninput=()=>{page=0;render();};$("prev").onclick=()=>{page--;render();};$("next").onclick=()=>{page++;render();};
    $("reset").onclick=()=>{for(const id of ["test","pw","discount","curvature","memory","clusters","quality","search"])$(id).value="";$("kind").value="model";$("hide-alias").checked=true;page=0;render();};
    $("clear-compare").onclick=()=>{selected.clear();render();renderComparison();};
    $("download-filtered").onclick=()=>download("filtered_diagnostics.csv",csv(["config_id","kind","restart_id","BIC","delta_BIC","LL","delta_LL","converged","probability_pass","boundary_count","LL_gap","path"],filtered.map(f=>[f.config_id,f.kind,f.restart_id,f.bic,f.delta_bic,f.ll,f.delta_ll,f.converged,f.probability_pass,f.boundary_count,f.ll_gap,f.path])),"text/csv;charset=utf-8");render();
  }
  function detail() {
    const d=DATA,s=d.summary,r=d.record,c=d.clusters||[];
    $("detail-title").textContent=s.config_id;$("detail-subtitle").textContent=`${s.path} · ${d.generated_at||""}`;
    if(d.error){document.querySelector("main").innerHTML=`<div class="error-view"><h2>这个文件未能转换</h2><pre>${esc(d.error)}</pre></div>`;return;}
    $("detail-cards").innerHTML=card("BIC",fmt(s.bic,2))+card("LL",fmt(s.ll,4))+card("类别／自由参数",`${s.spec.C??"—"} / ${s.k??"—"}`)+card("迭代数",fmt(s.n_iter,0))+card("概率检查",s.probability_pass===true?"通过":s.probability_pass===false?"不通过":"未完全核对")+card("ξ 中位数",fmt(s.ksi_median,5));
    $("detail-warning").textContent=(s.warnings?.length?s.warnings.join("；")+"。":"保存诊断未检出上述警示。")+" 不代表全局最优；未提供标准误、置信区间及样本外验证。";
    if(s.warnings?.length)$("detail-warning").classList.add("warn");
    if(!c.length){document.querySelector("[data-panel=parameters]").innerHTML="<p>非标准模型对象，完整内容见原始字段。</p>";}
    const rendered=new Set();
    function parameters() {
      if(!c.length)return;
      $("cluster-overview").innerHTML=table(["类别","π","后验软人数","MAP 人数",...RP,"概率参数","概率单调","实际负权重状态","A(100)"],c.map(q=>[esc(q.label),pct(q.pi),fmt(q.soft_n,2),fmt(q.map_n,0),...q.weights.map(pct),fmt(q.probability_parameter,6),q.monotone?"是":"否",fmt(q.negative_states,0),fmt(q.amplitude_100,6)]));
      $("parameter-table").innerHTML=table(["类别","参数","值","估计状态","下界","上界","边界警示"],c.flatMap(q=>q.parameters.map(p=>[esc(q.label),esc(p.name+(["s1","s2","s3"].includes(p.name)?"（内部编码）":"")),fmt(p.value,9),esc(p.status),fmt(p.lower,6),fmt(p.upper,6),p.boundary?badge(p.boundary,"warn"):"—"])));
      const W=680,H=c.length*43+40,svg=svgElement("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"参考点组合权重"});$("reference-chart").innerHTML='<div class="chart"><h3>每类的参考点组合权重</h3></div>';$("reference-chart").firstChild.appendChild(svg);
      c.forEach((q,i)=>{let x=90;svg.appendChild(svgElement("text",{x:80,y:i*43+25,"text-anchor":"end"},q.label));q.weights.forEach((w,j)=>{const rect=svgElement("rect",{x,y:i*43+8,width:Math.max(0,w)*520,height:28,fill:COLORS[j]});rect.appendChild(svgElement("title",{},`${q.label} · ${RP[j]}: ${pct(w)}`));svg.appendChild(rect);if(w>.08)svg.appendChild(svgElement("text",{x:x+w*260,y:i*43+27,"text-anchor":"middle",style:"fill:white"},`${RP[j]} ${pct(w)}`));x+=w*520;});});
      const probability=c.map((q,i)=>({name:`${q.label} (${q.method}=${fmt(q.probability_parameter,5)})`,color:COLORS[i],points:Array.from({length:201},(_,j)=>[j/200,weighting(j/200,q)])}));
      probability.push({name:"线性 w(p)=p",points:[[0,0],[1,1]],color:"#aab9c4",dashed:true});lineChart("probability-chart","累计概率权重函数",probability,{xmin:0,xmax:1,ymin:0,ymax:1,xlabel:"原累计概率 p"});
      function amount() {
        const value=Number($("amount").value);if(!(value>0)&&Number.isFinite(value))return;
        if(!Number.isFinite(value)||value<=0)return;$("amount-live").textContent=`£${fmt(value,2)}`;
        $("amount-table").innerHTML=table(["类别",`A(£${fmt(value,2)})`,`M(£${fmt(value,2)})`,"总幅度比较","边际比较"],c.map(q=>{const v=q.values,A=v.lambda*value**(v.alpha_loss-v.alpha_gain),M=A*v.alpha_loss/v.alpha_gain;return [esc(q.label),fmt(A,7),fmt(M,7),A>1?"损失较大":"收益较大／相等",M>1?"损失较敏感":"收益较敏感／相等"];}));
        lineChart("amount-chart","金额依赖：log10 A(d)，0 代表等额损益总幅度相同",ratioSeries(c),{xlabel:"相对参考点的金额 d（英镑，对数刻度）",xformat:x=>`£${fmt(10**x,2)}`,zero:true});
      }
      $("amount").oninput=()=>{$("amount-range").value=Math.log10(Math.max(.01,Number($("amount").value)||.01));amount();};
      $("amount-range").oninput=()=>{$("amount").value=(10**Number($("amount-range").value)).toFixed(2);amount();};amount();
    }
    function convergence() {
      const logs=r.iter_log||[],o=r.optimizer_log||[],g=d.diagnostics;let page=0;
      function curves() {
        const window=$("iteration-window").value,last=window==="all"?logs:logs.slice(-Number(window));
        const ll=last.map(row=>[row.iteration,row.log_likelihood]);if(ll.length&&r.log_likelihood!==ll[ll.length-1][1])ll.push([ll[ll.length-1][0],r.log_likelihood]);
        lineChart("ll-chart","LL：越高越好（末端含终值复算）",[{name:"LL",points:ll,color:"#127e7a"}],{xlabel:"EM 迭代"});
        lineChart("pi-chart","保存的类别份额轨迹",c.map((q,i)=>({name:q.label,color:COLORS[i],points:last.map(row=>[row.iteration,row.pis?.[i]])})),{ymin:0,ymax:1,xlabel:"EM 迭代",yformat:pct});
      }
      $("iteration-window").onchange=curves;curves();
      $("convergence-metrics").innerHTML=table(["指标","结果"],[["真实退出原因",esc(s.exit_reason)],["保存 LL",fmt(s.ll,9)],["终值复算改变",fmt(g.final_recomputation_change,9)],["最后两次日志 LL 差",fmt(g.last_logged_ll_change,9)],["LL 下降次数（阈值 −10⁻⁶）",fmt(g.ll_decreases,0)],["优化器调用数",fmt(g.optimizer_calls,0)],["优化器非成功退出",fmt(g.optimizer_unsuccessful,0)],["拒绝更新",fmt(g.optimizer_rejected,0)],["负Q目标增加次数",fmt(g.optimizer_q_increases,0)]]);
      $("iteration-table").innerHTML=table(["迭代","LL","日志 improvement","π","软人数","该轮秒数"],logs.map(row=>[fmt(row.iteration,0),fmt(row.log_likelihood,8),finite(row.improvement)?fmt(row.improvement,9):esc(row.improvement),(row.pis||[]).map(pct).join(" / "),(row.soft_n||[]).map(v=>fmt(v,2)).join(" / "),fmt(row.elapsed,4)]));
      $("optimizer-summary").textContent="表中 q_before/q_after 是负Q目标的一部分，越小越好；被拒绝时保留旧参数。";
      function optimizers() {
        const rows=o.filter(row=>!$("optimizer-only-bad").checked||!row.success||!row.accepted||(finite(row.q_after)&&row.q_after>row.q_before+1e-8)),pages=Math.max(1,Math.ceil(rows.length/40));page=Math.min(page,pages-1);
        $("optimizer-table").innerHTML=table(["迭代","类别","成功","接受更新","负Q前","负Q后","后−前","nfev","状态","完整消息"],rows.slice(page*40,(page+1)*40).map(row=>[fmt(row.iteration,0),fmt(row.cluster+1,0),row.success?"是":badge("否","warn"),row.accepted?"是":badge("拒绝","warn"),fmt(row.q_before,7),fmt(row.q_after,7),fmt(row.q_after-row.q_before,7),fmt(row.nfev,0),esc(row.status),esc(row.message)]));
        $("optimizer-page").textContent=`${rows.length} 条 · ${page+1}/${pages}`;$("optimizer-prev").disabled=page===0;$("optimizer-next").disabled=page===pages-1;
      }
      $("optimizer-only-bad").onchange=()=>{page=0;optimizers();};$("optimizer-prev").onclick=()=>{page--;optimizers();};$("optimizer-next").onclick=()=>{page++;optimizers();};optimizers();
      if(r.restart_summaries?.length)$("restart-summaries").innerHTML="<h3>该模型保存的全部起点摘要</h3>"+table(["起点","LL","距本摘要最高 LL","达到 LL 容差","迭代","退出原因","运行秒数"],r.restart_summaries.map(row=>[esc(row.restart_id),fmt(row.log_likelihood,6),fmt(Math.max(...r.restart_summaries.map(q=>q.log_likelihood))-row.log_likelihood,6),row.converged?"是":"否",fmt(row.n_iter,0),esc(row.exit_reason),fmt(row.elapsed,3)]));
    }
    function fit() {
      const f=d.fit;if(f.status!=="verified"){$("fit-status").textContent=f.reason||"未进行当前数据复算";$("fit-status").classList.add("warn");return;}
      $("fit-status").textContent="使用当前数据和与协议一致的共享标量估值公式，复算最终 CE、LL、后验以及实际条件彩票的决策权重。";
      $("fit-metrics").innerHTML=table(["指标","结果"],[["复算 LL",fmt(f.ll_recomputed,9)],["复算−保存 LL",fmt(f.ll_gap,9)],["后验最大绝对差",fmt(f.posterior_gap,9)],["全样本 RMSE",fmt(f.rmse,5)],["全样本归一化 RMSE",fmt(f.normalized_rmse,7)]]);
      lineChart("fit-chart","分阶段归一化 RMSE",[{name:"归一化 RMSE",points:f.stage.map(q=>[q.stage,q.normalized_rmse]),color:"#127e7a"}],{xlabel:"估值阶段 t"});
      lineChart("fit-bias-chart","分阶段平均误差：观测−拟合",[{name:"平均误差（£）",points:f.stage.map(q=>[q.stage,q.bias]),color:"#d49436"}],{xlabel:"估值阶段 t",zero:true});
      $("stage-table").innerHTML=table(["阶段","观测数","平均误差 £","RMSE £","归一化 RMSE"],f.stage.map(q=>[fmt(q.stage,0),fmt(q.n,0),fmt(q.bias,5),fmt(q.rmse,5),fmt(q.normalized_rmse,7)]));
      $("state-table").innerHTML=table(["彩票","t","z1","z2","CE 数","观测均值","条件剩余期望",...c.flatMap(q=>[`${q.label} 参考点`,`${q.label} 理论 CE`])],f.states.map(q=>[esc(q.lottery),fmt(q.stage,0),fmt(q.z1,2),fmt(q.z2,2),fmt(q.n,0),fmt(q.observed_mean,5),fmt(q.conditional_expectation,5),...c.flatMap((_,i)=>[fmt(q.reference[i],6),fmt(q.ce[i],6)])]));
    }
    function people() {
      const subjects=r.subjects||[],ksi=r.ksi||[],resp=r.resp||[];let page=0,people=[];
      histogram("ksi-chart","全部个人 ξ 的分布",ksi);if(c.length===1)emptyChart("posterior-chart","最大后验概率","单类模型：后验 1 是模型限制，不能作为分类准确率");else histogram("posterior-chart","每人的最大类别后验",resp.map(row=>Math.max(...row)));
      function render() {
        const text=$("person-search").value.toLowerCase();people=subjects.map((id,i)=>({id,i})).filter(q=>String(q.id).toLowerCase().includes(text));const pages=Math.max(1,Math.ceil(people.length/25));page=Math.min(page,pages-1);
        $("people-table").innerHTML=table(["参与者","ξ",...c.map(q=>`${q.label} 后验`),"MAP 类别","最大后验","熵"],people.slice(page*25,(page+1)*25).map(({id,i})=>{const row=resp[i],max=Math.max(...row),index=row.indexOf(max),entropy=-row.reduce((a,p)=>a+p*Math.log(Math.max(p,1e-300)),0);return [esc(id),fmt(ksi[i],9),...row.map(p=>fmt(p,9)),esc(c[index].label),fmt(max,9),fmt(entropy,9)];}));
        $("people-page").textContent=`${people.length} 人 · ${page+1}/${pages}`;$("people-prev").disabled=page===0;$("people-next").disabled=page===pages-1;
      }
      $("person-search").oninput=()=>{page=0;render();};$("people-prev").onclick=()=>{page--;render();};$("people-next").onclick=()=>{page++;render();};
      $("download-people").onclick=()=>download(`${s.config_id}-people.csv`,csv(["subject","ksi",...c.map(q=>`posterior_${q.cluster}`)],subjects.map((id,i)=>[id,ksi[i],...resp[i]])),"text/csv;charset=utf-8");render();
    }
    function raw() {
      const keys=r&&typeof r==="object"?Object.keys(r):[];$("raw-field").innerHTML='<option value="__all__">全部原始字段</option>'+keys.map(key=>`<option value="${esc(key)}">${esc(key)}</option>`).join("");
      const show=()=>{$("raw-view").textContent=JSON.stringify($("raw-field").value==="__all__"?r:r[$("raw-field").value],null,2);};
      $("raw-field").onchange=show;if(keys.length)$("raw-field").value=keys.includes("spec")?"spec":keys[0];show();$("meta-view").textContent=JSON.stringify(d.metadata||{},null,2);
      $("download-json").onclick=()=>download(`${s.config_id}-${r?.restart_id||"result"}.json`,JSON.stringify(d,null,2),"application/json;charset=utf-8");
    }
    const renderers={parameters,convergence,fit,people,raw};
    document.querySelectorAll("#detail-tabs [data-tab]").forEach(button=>button.onclick=()=>{const name=button.dataset.tab;document.querySelectorAll("[data-panel]").forEach(p=>p.hidden=p.dataset.panel!==name);document.querySelectorAll("#detail-tabs button").forEach(b=>b.classList.toggle("active",b===button));if(!rendered.has(name)){renderers[name]();rendered.add(name);}});
    if(c.length){parameters();rendered.add("parameters");}
    else{document.querySelectorAll("[data-panel]").forEach(p=>p.hidden=p.dataset.panel!=="raw");document.querySelectorAll("#detail-tabs button").forEach(b=>{b.disabled=b.dataset.tab!=="raw";b.classList.toggle("active",b.dataset.tab==="raw");});raw();rendered.add("raw");}
  }
  if(document.body.dataset.page==="dashboard")dashboard();else detail();
})();
