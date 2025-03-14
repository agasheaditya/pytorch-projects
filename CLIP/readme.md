# **Text2Art**: Transforming words into masterpieces
---

### **1. Aim**
The aim of this project is to build a text-driven neural style transfer model using CLIP (Contrastive Language–Image Pretraining) and a U-Net-based generator. The goal is to:

Generate styled images based on a content image and a text prompt.
Align the generated image with the artistic style described in the text prompt.
Evaluate the model's performance and explore its potential applications in creative domains like art generation, design, and visual storytelling.
### **2. Problem Description**
Neural Style Transfer (NST) traditionally involves transferring the style of one image onto another while preserving the content of the original image. However, traditional NST methods rely on fixed style images , limiting their flexibility.

In this project, we address the following challenges:

  - Text-to-Style Mapping : How to map textual descriptions (e.g., "Van Gogh's Starry Night") into visual styles.
  - Cross-Modal Learning : How to align textual embeddings (from CLIP) with visual features (from images).
  - Training Stability : Ensuring the model learns meaningful representations without overfitting or mode collapse.
  - Generalization : Generating high-quality styled images for diverse text prompts and content images.

### **3. Proposed Solution**
To solve the problem, we propose the following approach:

CLIP Embeddings : Use CLIP to encode both images and text into a shared embedding space. This allows us to compute similarity between text prompts and images.
U-Net Generator : Use a U-Net architecture to generate styled images by combining content images and text embeddings.
Loss Functions :
Content Loss : Ensures the generated image retains the structure of the content image (using VGG features).
Style Loss : Measures the cosine similarity between the CLIP embeddings of the generated image and the text prompt.
Training Pipeline :
Train the generator to minimize a weighted combination of content loss and style loss.
Inference :
Use the trained generator to generate styled images from new content images and text prompts.

### **4. Training Data Information**

```md
wikiart_subset/
│
├── Cubism/
│       Cubism-0.jpg
│       Cubism-1.jpg
│       ...
├── Fauvism/
│       Fauvism-0.jpg
│       Fauvism-1.jpg
│       ...
├── Minimalism/
│       Minimalism-0.jpg
│       Minimalism-1.jpg
│       ...
└── Ukiyo_e/
        Ukiyo_e-0.jpg
        Ukiyo_e-1.jpg
        ...
```
#### **Metadata** : Each image has a corresponding text description stored in image_labels.json. For example:

```md
{
    "Cubism/Cubism-0.jpg": "An urban scene reimagined through fragmented perspectives and interlocking shapes.",
    "Fauvism/Fauvism-0.jpg": "An artwork defined by energetic brushstrokes and a focus on color over detail."
}
```

#### **Preprocessing**
Images are resized to 256x256 pixels.
Images are normalized using the ImageNet mean ([0.485, 0.456, 0.406]) and standard deviation ([0.229, 0.224, 0.225]).
Text descriptions are tokenized and encoded into CLIP embeddings.

### **5. Data Flow Diagram and Explanation**
Data Flow Diagram

```md
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Content Image    +------>+  Normalize        +------>+  Encode Image     |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
                                                                 |
                                                                 v
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Text Prompt      +------>+  Tokenize         +------>+  Encode Text      |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
                                                                 |
                                                                 v
                                                     +-----------------------+
                                                     |                       |
                                                     |  Combine Embeddings   |
                                                     |  (Generator Input)    |
                                                     |                       |
                                                     +-----------------------+
                                                                 |
                                                                 v
                                                     +-----------------------+
                                                     |                       |
                                                     |  U-Net Generator      |
                                                     |  (Styled Image)       |
                                                     |                       |
                                                     +-----------------------+
```
**Explanation**
Content Image :
Load and preprocess the content image (resize, normalize).
Encode the image into a CLIP embedding.
Text Prompt :
Tokenize the text prompt and encode it into a CLIP embedding.
Combine Embeddings :
Pass the content image and text embedding into the U-Net generator.
The generator combines the two inputs to produce a styled image.
Output :
The styled image is denormalized and converted back to a PIL image for visualization.
--- 
### **6. Working of the Code Along with Algorithm/Model Used**
Algorithm
Input :
Content image: content_img.
Text prompt: text_prompt.
Preprocessing :
Normalize the content image.
Encode the text prompt into a CLIP embedding.
Forward Pass :
Pass the content image and text embedding into the U-Net generator.
Generate the styled image.
Loss Computation :
Compute content loss (VGG features).
Compute style loss (CLIP embeddings).
Combine losses using weights (alpha and beta).
Backpropagation :
Update the generator weights using gradient descent.
Inference :
Generate styled images for new inputs.

### **7. Inference**
After training, the model can generate styled images from any content image and text prompt. For example:

Content Image : A photograph of a cityscape.
Text Prompt : "Van Gogh's Starry Night with swirling blue skies and bright yellow stars."
Output : A stylized version of the cityscape that resembles Van Gogh's painting.

### **8. Achievements**
Successfully implemented a text-driven neural style transfer pipeline.
Demonstrated the ability to generate styled images from diverse text prompts.
Built a modular and reusable codebase for future experiments.

### **9. Future Scope and Improvements**
Architecture Enhancements :
Replace U-Net with more advanced architectures like AdaIN or StyleGAN.
Add attention mechanisms to better capture fine-grained details.
Dataset Expansion :
Use larger datasets (e.g., LAION-Aesthetics) to improve generalization.
Evaluation Metrics :
Use metrics like FID (Fréchet Inception Distance) to evaluate image quality.
Interactive Interface :
Build a web app or GUI for users to upload images and enter text prompts.
Multi-Modal Inputs :
Extend the model to handle multiple text prompts or style references.
