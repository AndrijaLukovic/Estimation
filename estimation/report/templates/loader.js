"use strict";
window.ReportLoader = (() => {
  async function payload(key, descriptor) {
    const fragments = [];
    window.ReportData = {add(id, index, fragment) { if(id === key) fragments[index] = fragment; }};
    for(const path of descriptor.parts) {
      await new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = path; script.onload = resolve;
        script.onerror = () => reject(new Error(`Unable to load ${path}`));
        document.head.appendChild(script);
      });
    }
    for(let index=0; index<descriptor.chunks; index++) {
      if(typeof fragments[index] !== "string") throw new Error("Incomplete report data");
    }
    return JSON.parse(fragments.join(""));
  }
  async function load() {
    const metadata = JSON.parse(document.getElementById("report-data").textContent);
    if(!metadata.storage) return metadata;
    if(document.body.dataset.page === "dashboard") {
      return {...metadata, files: await payload("catalog", metadata.storage.catalog)};
    }
    const id = new URLSearchParams(window.location.search).get("id");
    const descriptor = metadata.storage.details[id];
    if(!descriptor) throw new Error("Unknown report record");
    return payload(id, descriptor);
  }
  return {load};
})();
