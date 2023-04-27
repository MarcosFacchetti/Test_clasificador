import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import modelJson from "./model.json";

function CameraClassification() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const otrocanvasRef = useRef(null);
  const resultadoRef = useRef(null);
  const [modelo, setModelo] = useState(null);
  let currentStream = null;
  let facingMode = "user";
  const tamano = 400;

  useEffect(() => {
    cargarModelo();
    mostrarCamara();
    return () => {
      if (currentStream) {
        currentStream.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  async function cargarModelo() {
    console.log("Cargando modelo...");
    const loadedModel = await tf.loadLayersModel(modelJson);
    setModelo(loadedModel);
    console.log("Modelo cargado");
  }

  function mostrarCamara() {
    const opciones = {
      audio: false,
      video: {
        width: tamano,
        height: tamano,
      },
    };

    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia(opciones)
        .then((stream) => {
          currentStream = stream;
          videoRef.current.srcObject = currentStream;
          procesarCamara();
          predecir();
        })
        .catch((err) => {
          alert("No se pudo utilizar la cámara :(");
          console.log(err);
          alert(err);
        });
    } else {
      alert("No existe la función getUserMedia");
    }
  }

  function cambiarCamara() {
    if (currentStream) {
      currentStream.getTracks().forEach((track) => {
        track.stop();
      });
    }

    facingMode = facingMode === "user" ? "environment" : "user";

    const opciones = {
      audio: false,
      video: {
        facingMode: facingMode,
        width: tamano,
        height: tamano,
      },
    };

    navigator.mediaDevices
      .getUserMedia(opciones)
      .then((stream) => {
        currentStream = stream;
        videoRef.current.srcObject = currentStream;
      })
      .catch((err) => {
        console.log("Oops, hubo un error", err);
      });
  }

  function procesarCamara() {
    const ctx = canvasRef.current.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, tamano, tamano, 0, 0, tamano, tamano);
    setTimeout(procesarCamara, 20);
  }


  function resample_single(canvas, width, height, resize_canvas) {
    var width_source = canvas.width;
    var height_source = canvas.height;
    width = Math.round(width);
    height = Math.round(height);
  
    var ratio_w = width_source / width;
    var ratio_h = height_source / height;
    var ratio_w_half = Math.ceil(ratio_w / 2);
    var ratio_h_half = Math.ceil(ratio_h / 2);
  
    var ctx = canvas.getContext("2d");
    var ctx2 = resize_canvas.getContext("2d");
    var img = ctx.getImageData(0, 0, width_source, height_source);
    var img2 = ctx2.createImageData(width, height);
    var data = img.data;
    var data2 = img2.data;
  
    for (var j = 0; j < height; j++) {
      for (var i = 0; i < width; i++) {
        var x2 = (i + j * width) * 4;
        var weight = 0;
        var weights = 0;
        var weights_alpha = 0;
        var gx_r = 0;
        var gx_g = 0;
        var gx_b = 0;
        var gx_a = 0;
        var center_y = (j + 0.5) * ratio_h;
        var yy_start = Math.floor(j * ratio_h);
        var yy_stop = Math.ceil((j + 1) * ratio_h);
        for (var yy = yy_start; yy < yy_stop; yy++) {
          var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
          var center_x = (i + 0.5) * ratio_w;
          var w0 = dy * dy; //pre-calc part of w
          var xx_start = Math.floor(i * ratio_w);
          var xx_stop = Math.ceil((i + 1) * ratio_w);
          for (var xx = xx_start; xx < xx_stop; xx++) {
            var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
            var w = Math.sqrt(w0 + dx * dx);
            if (w >= 1) {
              //pixel too far
              continue;
            }
            //hermite filter
            weight = 2 * w * w * w - 3 * w * w + 1;
            var pos_x = 4 * (xx + yy * width_source);
            //alpha
            gx_a += weight * data[pos_x + 3];
            weights_alpha += weight;
            //colors
            if (data[pos_x + 3] < 255)
              weight = weight * data[pos_x + 3] / 250;
            gx_r += weight * data[pos_x];
            gx_g += weight * data[pos_x + 1];
            gx_b += weight * data[pos_x + 2];
            weights += weight;
          }
        }
        data2[x2] = gx_r / weights;
        data2[x2 + 1] = gx_g / weights;
        data2[x2 + 2] = gx_b / weights;
        data2[x2 + 3] = gx_a / weights_alpha;
      }
    }
  
    ctx2.putImageData(img2, 0, 0);
  }
  


  function predecir() {
    if (modelo !== null) {
      resample_single(canvasRef.current, 100, 100, otrocanvasRef.current);
  
      // Hacer la predicción
      const ctx2 = otrocanvasRef.current.getContext("2d");
      const imgData = ctx2.getImageData(0, 0, 100, 100);
  
      const arr = [];
      let arr100 = [];
  
      for (let p = 0; p < imgData.data.length; p += 4) {
        const rojo = imgData.data[p] / 255;
        const verde = imgData.data[p + 1] / 255;
        const azul = imgData.data[p + 2] / 255;
  
        const gris = (rojo + verde + azul) / 3;
  
        arr100.push([gris]);
        if (arr100.length === 100) {
          arr.push(arr100);
          arr100 = [];
        }
      }
  
      const tensor = tf.tensor4d([arr]);
      const resultado = modelo.predict(tensor).dataSync();
  
      let respuesta;
      if (resultado <= 0.5) {
        respuesta = "Gato";
      } else {
        respuesta = "Perro";
      }
      resultadoRef.current.innerHTML = respuesta;
    }
  
    setTimeout(predecir, 150);
  }
  

  return (
    <div>
      <div>
        <video
          ref={videoRef}
          width={tamano}
          height={tamano}
          style={{ transform: "scaleX(-1)" }}
          playsInline
          autoPlay
          muted
        ></video>
        <canvas ref={canvasRef} width={tamano} height={tamano}></canvas>
        <canvas
          ref={otrocanvasRef}
          width={100}
          height={100}
          style={{ display: "none" }}
        ></canvas>
      </div>
      <div>
        <button onClick={cambiarCamara}>Cambiar cámara</button>
        <div ref={resultadoRef}></div>
      </div>
    </div>
  );
}

export default CameraClassification;


  