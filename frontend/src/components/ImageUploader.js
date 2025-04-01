import { useState } from "react";
import "./ImageUploader.css"; // Keep styles separate

const ImageUploader = () => {
    const [dataset, setDataset] = useState("");
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [batchNormImage, setBatchNormImage] = useState(null);
    const [gradCamImage, setGradCamImage] = useState(null);
    const [loadingModel, setLoadingModel] = useState(false);
    const [modelLoaded, setModelLoaded] = useState(false);
    const [processingImage, setProcessingImage] = useState(false);
    const [error, setError] = useState(null);

    const handleDatasetChange = (e) => setDataset(e.target.value);
    const handleImageChange = (e) => setImage(e.target.files[0]);

    const loadModel = async () => {
        if (!dataset) return alert("Please select a dataset!");

        setLoadingModel(true);
        setModelLoaded(false);
        setError(null);

        try {
            const response = await fetch("http://127.0.0.1:5000/load_model", {  // Change URL to local
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dataset_name: dataset }),
            });

            if (!response.ok) {
                throw new Error(`Failed to load model: ${response.statusText}`);
            }

            const data = await response.json();
            setLoadingModel(false);
            setModelLoaded(true);

            alert(data.message || data.error);
        } catch (error) {
            setLoadingModel(false);
            setError(error.message);
        }
    };

    const predictImage = async () => {
        if (!image) return alert("Please select an image!");

        setPrediction(null);
        setBatchNormImage(null);
        setGradCamImage(null);
        setProcessingImage(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", image);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {  // Change URL to local
                method: "POST", 
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.statusText}`);
            }

            const data = await response.json();
            setProcessingImage(false);

            // Display "Fire" if prediction is 1, "No Fire" if prediction is 0
            setPrediction(data.predicted_class === 1 ? "🔥 Fire" : "✅ No Fire");

            const timestamp = new Date().getTime(); // Force image refresh
            setBatchNormImage(`http://127.0.0.1:5000/uploads/batch_norm_vis.png?t=${timestamp}`);  // Local URL for BatchNorm
            setGradCamImage(`http://127.0.0.1:5000/uploads/gradcam_vis.png?t=${timestamp}`);  // Local URL for GradCam

        } catch (error) {
            setProcessingImage(false);
            setError(error.message);
        }
    };

    return (
        <div>
            <h2>Select Dataset & Upload Image</h2>
            <select onChange={handleDatasetChange} value={dataset}>
                <option value="">Select Dataset</option>
                <option value="CairFire">CairFire</option>
                <option value="FireDetection">FireDetection</option>
                <option value="FireNet">FireNet</option>
                <option value="FireSense">FireSense</option>
                <option value="FireSmoke">FireSmoke</option>
                <option value="ThermalFire">ThermalFire</option>
            </select>
            <button onClick={loadModel} disabled={loadingModel}>Load Model</button>

            {/* Model Loading Status */}
            {loadingModel && <p className="loading-text">Loading model...</p>}
            {modelLoaded && !loadingModel && <p className="success-text">Model loaded successfully!</p>}

            <input type="file" onChange={handleImageChange} />
            <button onClick={predictImage} disabled={processingImage || !modelLoaded}>Predict</button>

            {/* Error Handling */}
            {error && <p className="error-text">{error}</p>}

            {/* Image Processing Status */}
            {processingImage && <p className="loading-text">Processing image...</p>}

            {prediction !== null && <h3 className="prediction-text">{prediction}</h3>}

            {batchNormImage && (
                <div>
                    <h3>Batch Norm Effect</h3>
                    <img src={batchNormImage} alt="Batch Norm Visualization" width="400px" />
                </div>
            )}

            {gradCamImage && (
                <div>
                    <h3>Grad-CAM Heatmap</h3>
                    <img src={gradCamImage} alt="Grad-CAM Visualization" width="400px" />
                </div>
            )}
        </div>
    );
};

export default ImageUploader;
