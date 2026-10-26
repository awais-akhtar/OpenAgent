import{c as o}from"../app.js";/**
 * @license lucide-react v1.56.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const n={name:"ellipsis",size:24,node:[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]],aliases:["more-horizontal"]};n.node;const s=o(n);function a(e){if(!["ArrowLeft","ArrowRight","Home","End"].includes(e.key))return;const c=Array.from(e.currentTarget.querySelectorAll('[role="tab"]'));if(!c.length)return;const r=c.indexOf(document.activeElement);if(r<0)return;e.preventDefault();const t=e.key==="Home"?0:e.key==="End"?c.length-1:(r+(e.key==="ArrowRight"?1:-1)+c.length)%c.length;c[t].focus(),c[t].click()}const y=e=>e?0:-1;export{s as E,a as h,y as t};
