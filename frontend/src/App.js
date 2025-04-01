import React from "react";
import ImageUploader from "./components/ImageUploader"; // Import the component
import "./App.css"; // Import global styles

function App() {
  return (
    <div className="App">
      <h1>Department of Information Technology</h1>
      <h2>National Institute of Technology, Karnataka, Surathkal</h2>
      <h1>CNN for Fire Classification</h1>
      <h2>Submitted by: Aakarsh Bansal (221AI001), Smruthi Bhat (221AI038) </h2>
      <h2>Work done as part of IT353 course project for the session from January - April 2025</h2>
      <ImageUploader /> {/* Render the ImageUploader component */}
    </div>
  );
}

export default App;
