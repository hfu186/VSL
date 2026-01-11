function initCWASA() {
  CWASA.init({
    avs: ["anna"],
    avSettings: [
      {
        width: 700,
        height: 600,
        avList: "avs",
        initAv: "anna",
        initCamera: [0, 0.23, 3.24, 5, 18, 30, -1, -1],
        allowFrameSteps: true,    
        allowSpeedControls: true,  
        allowSiGMLText: true,
        rateSpeed: 1  
      }
    ]
  });
}
