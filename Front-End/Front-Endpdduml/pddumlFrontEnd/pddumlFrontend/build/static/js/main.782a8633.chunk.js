(window.webpackJsonp=window.webpackJsonp||[]).push([[0],{10:function(e,t,a){},14:function(e,t,a){e.exports=a(35)},23:function(e,t,a){},25:function(e,t,a){},35:function(e,t,a){"use strict";a.r(t);var n=a(1),l=a.n(n),c=a(12),r=a.n(c),o=(a(23),a(3)),i=(a(25),a(37));var u=function(){var e=Object(n.useRef)(null),t=Object(n.useRef)(null),a=Object(n.useState)(!1),c=Object(o.a)(a,2),r=c[0],i=c[1],u=Object(n.useState)("user"),m=Object(o.a)(u,2),s=m[0],d=m[1],p=Object(n.useState)(!1),E=Object(o.a)(p,2),v=E[0],b=E[1],g=Object(n.useState)(" "),f=Object(o.a)(g,2),h=(f[0],f[1],Object(n.useState)("")),j=Object(o.a)(h,2),O=j[0],C=(j[1],Object(n.useState)("")),S=Object(o.a)(C,2),y=(S[0],S[1]),N=Object(n.useState)([]),w=Object(o.a)(N,2);return w[0],w[1],Object(n.useEffect)(function(){navigator.mediaDevices.getUserMedia({video:{width:256,height:256,facingMode:s}}).then(function(t){var a=e.current;a.srcObject=t,a.play()}).catch(function(e){console.error(e)})},[e,s]),l.a.createElement("div",{className:"CameraApp"},l.a.createElement("div",{className:"box"},l.a.createElement("h3",null,O)),l.a.createElement("div",{className:"Camera"},l.a.createElement("video",{ref:e}),l.a.createElement("div",{className:"result"+(r?"hasPhoto":"")}),l.a.createElement("canvas",{id:"myCanvas",ref:t}),l.a.createElement("p",null,v?l.a.createElement("button",{className:"btn",onClick:function(){var e=t.current;e.getContext("2d").clearRect(0,0,e.width,e.height),i(!1),b(!1)}},"Recapture"):l.a.createElement("button",{className:"btn",onClick:function(){document.getElementById("myCanvas");var a=e.current,n=t.current;n.width=256,n.height=256;var l=n.getContext("2d");l.drawImage(a,0,0,256,256);var c=l.getImageData(0,0,n.width,n.height).data,r=[],o=[];console.log(c.length);for(var u=0,m=0;m<c.length;m+=4){u%256===0&&u>0&&(r.push(o),o=[]);var s=c[m],d=c[m+1],p=c[m+2];u+=1,o.push([s,d,p])}r.push(o),y(r),i(!0),b(!0)}},"Capture")),l.a.createElement("p",null,l.a.createElement("button",{className:"btn",onClick:function(){d("user"===s?"environment":"user")}},"Switch Camera")),l.a.createElement("p",null,l.a.createElement("button",{className:"btn",onClick:function(){!function(e,t){var a=document.getElementById(e);console.log(e);var n=a.toDataURL("image/jpeg"),l=document.createElement("a");l.href=n,l.download=t,document.body.appendChild(l),l.click(),document.body.removeChild(l)}("myCanvas","myImage.jpeg"),console.log(t)}},"Download"))))},m=(a(10),a(13));var s=function(){var e=Object(n.useState)(" "),t=Object(o.a)(e,2),a=t[0],c=t[1],r=Object(n.useState)([]),u=Object(o.a)(r,2),s=u[0],d=u[1],p=Object(n.useState)(null),E=Object(o.a)(p,2),v=(E[0],E[1],Object(n.useState)("")),b=Object(o.a)(v,2),g=b[0],f=b[1],h=Object(n.useState)(" "),j=Object(o.a)(h,2),O=(j[0],j[1],Object(n.useState)("")),C=Object(o.a)(O,2),S=C[0],y=C[1];return l.a.createElement("div",{className:"App"},l.a.createElement("div",{className:"Form"},l.a.createElement("div",{className:"box"},l.a.createElement("h3",null,g),l.a.createElement("img",{height:250,width:250,src:a})),l.a.createElement("div",{className:"box"},l.a.createElement("form",null,l.a.createElement("p",null,l.a.createElement("strong",null,"Upload Image of leaf:              "),l.a.createElement("input",{type:"file",name:"imagefile",id:"imgfile",onChange:function(e){return function(e){f(""),console.log(s.length);var t=e.target.files[0];c(URL.createObjectURL(e.target.files[0]));var a=new FileReader;a.addEventListener("load",function(e){for(var t=e.target.result,a=Object(m.decode)(new Uint8Array(t),{useTArray:!0}),n=[],l=[],c=0;c<a.height;c++){for(var r=[],o=0;o<a.width;o++){var i=4*(c*a.width+o),u=a.data[i],s=a.data[i+1],p=a.data[i+2];a.data[i+3],n.push(p),r.push([u,s,p])}l.push(r)}d(n),y(l)}),a.readAsArrayBuffer(t)}(e)},required:!0})),l.a.createElement("p",null,l.a.createElement("button",{className:"btn",value:"Submit",onClick:function(){i.a.post("http://127.0.0.1:8000/gettomoato/",{CNNImg:S}).then(function(e){console.log(e.data),alert("Plant - "+e.data.plantname+"\nDisease - "+e.data.disease+"\nConfidence - "+e.data.confidence+"\n"+e.data.comments)})}},"Submit")),l.a.createElement("p",null,l.a.createElement("button",{className:"btn",value:"Reset"},"Reset"))))))};var d=function(){var e=Object(n.useState)(!0),t=Object(o.a)(e,2),a=t[0],c=t[1];return l.a.createElement("div",{className:"App"},l.a.createElement("div",{className:"Tog"},l.a.createElement("button",{className:"btnHead",onClick:function(){return c(!a)}},"Upload or Capture")),a?l.a.createElement(s,null):l.a.createElement("div",{className:"Cam"},l.a.createElement(u,null)," "))};var p=function(){return l.a.createElement("div",{class:"A"},l.a.createElement("div",{class:"container"},l.a.createElement("h1",{class:"display-4"},"Plant Disease Detection Using ML"),l.a.createElement("p",{class:"lead"},"Upload Image or Capture a image to know the Disease")))};var E=function(){var e=Object(n.useState)(" "),t=Object(o.a)(e,2),a=(t[0],t[1]),c=Object(n.useState)(0),r=Object(o.a)(c,2),u=(r[0],r[1]);return l.a.createElement("div",{className:"App"},l.a.createElement("div",{class:"Form"},l.a.createElement("strong",null,"Feedback Form for Incorrectly Classified Images"),l.a.createElement("form",null,l.a.createElement("p",null,l.a.createElement("strong",null,"Upload Incorrectly Classified Image of leaf:              "),l.a.createElement("input",{type:"file",name:"imagefile",id:"imgfile",onChange:function(e){return u(e.target.value)},required:!0})),l.a.createElement("p",null,l.a.createElement("strong",null,"Select Identified Disease:                      "),l.a.createElement("select",{name:"plant",id:"plant",onChange:function(e){return a(e.target.value)},required:!0},l.a.createElement("option",{value:""},"Tomato Bacterial Spots"),l.a.createElement("option",{value:""},"Papper Bell Bacterial Spots"),l.a.createElement("option",{value:""},"Potato Bacterial Spots"),l.a.createElement("option",{value:""},"Tomato Early Blight"),l.a.createElement("option",{value:""},"Potato Early Blight"),l.a.createElement("option",{value:""},"Tomato Late Blight"),l.a.createElement("option",{value:""},"Potato Late Blight"),l.a.createElement("option",{value:""},"Tomato Healthy"),l.a.createElement("option",{value:""},"Papper Bell Healthy "),l.a.createElement("option",{value:""},"Potato Healthy"))),l.a.createElement("p",null,l.a.createElement("input",{type:"submit",value:"Submit",onClick:function(){i.a.post("http://127.0.0.1:8000/test/",{data:0}).then(function(e){return console.log(e.data)})}})),l.a.createElement("p",null,l.a.createElement("input",{type:"reset",value:"Reset"})))))},v=function(e){e&&e instanceof Function&&a.e(1).then(a.bind(null,36)).then(function(t){var a=t.getCLS,n=t.getFID,l=t.getFCP,c=t.getLCP,r=t.getTTFB;a(e),n(e),l(e),c(e),r(e)})};r.a.createRoot(document.getElementById("root")).render(l.a.createElement(l.a.StrictMode,null,l.a.createElement(p,null),l.a.createElement(d,null),l.a.createElement(E,null))),v()}},[[14,3,2]]]);
//# sourceMappingURL=main.782a8633.chunk.js.map