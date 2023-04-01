const modelUrl = 'https://raw.githubusercontent.com/Miaad2004/From-Edges-to-Cats-Pix2Pix-Image-Translation-in-TensorFlow/main/WebApp/model-TFJS/model.json';

let gray = '#1c1c1b';
let white = '#ffffff';
let drawingColor = gray;
let drawingRadius = 0.5;

// Get the canvas elements
const inputCanvas = document.getElementById('input-canvas');
const outputCanvas = document.getElementById('output-canvas');

const inputCTX = inputCanvas.getContext('2d');
const outputCTX = outputCanvas.getContext('2d');

// Set the canvas size 
inputCanvas.width = 256;
inputCanvas.height = 256;

outputCanvas.width = 256;
outputCanvas.height = 256;

// Drawing handler
let isDrawing = false;

inputCanvas.addEventListener('mousedown', startDrawing);
inputCanvas.addEventListener('mousemove', draw);
inputCanvas.addEventListener('mouseup', endDrawing);

// Touch event handlers
inputCanvas.addEventListener('touchstart', startDrawing);
inputCanvas.addEventListener('touchmove', drawTouch);
inputCanvas.addEventListener('touchend', endDrawing);


function swapDrawingColor()
{
  if (drawingColor == gray)
  {
    drawingColor = white;
    drawingRadius = 3;
  }

  else
  {
    drawingColor = gray;
    drawingRadius = 0.9;
  }
}

function startDrawing(e)
 {
  isDrawing = true;
  draw(e);
}

function drawTouch(e) {
  e.preventDefault();
  const touch = e.touches[0];
  const canvasRect = inputCanvas.getBoundingClientRect();
  const offsetX = touch.clientX - canvasRect.left;
  const offsetY = touch.clientY - canvasRect.top;
  draw({ offsetX, offsetY });
}

function draw(e) 
{
  if (!isDrawing) 
  {
    return;
  }

  // Draw a circle at the current position
  inputCTX.beginPath();
  inputCTX.arc(e.offsetX, e.offsetY, drawingRadius, 0, 2 * Math.PI);
  inputCTX.fillStyle = drawingColor;
  inputCTX.fill();
}

function endDrawing()
 {
  isDrawing = false;
}

// Add default images randomly
const randomButton = document.getElementById('randomButton');
randomButton.addEventListener('click', setCanvasImageRandomly);

function setCanvasImage(ctx, imageSrc)
{
  const image = new Image();
  image.src = imageSrc;
  image.onload = function()
  {
    ctx.drawImage(image, 0, 0, 256, 256);
  }
}

function setCanvasImageRandomly()
{
  const numSamples = 34;
  var randomIndex = Math.floor(Math.random() * numSamples);
  randomIndex = String(randomIndex).padStart(2, '0');

  const inputImageUrl = `https://raw.githubusercontent.com/Miaad2004/From-Edges-to-Cats-Pix2Pix-Image-Translation-in-TensorFlow/main/WebApp/samples/i${randomIndex}.jpg`;
  const outputImageUrl = `https://raw.githubusercontent.com/Miaad2004/From-Edges-to-Cats-Pix2Pix-Image-Translation-in-TensorFlow/main/WebApp/samples/o${randomIndex}.jpg`;

  setCanvasImage(inputCTX, inputImageUrl);
  setCanvasImage(outputCTX, outputImageUrl);
}

setCanvasImageRandomly();

// Image denormalizer
function denormalize(image) 
{
  const denormalized = image.mul(127.5).add(127.5).cast('int32');
  return denormalized;
}

function normalize(image) 
{
  const normalized = image.sub(127.5).div(127.5).cast('float32');
  return normalized;
}

function disableGenerateButton(text)
{
  generateButton.disabled = true;
  generateButton.classList.add('disabled');
  generateButton.textContent = text;
}

function enableGenerateButton()
{
  generateButton.disabled = false;
  generateButton.classList.remove('disabled');
  generateButton.textContent = 'Generate';
}

async function loadModel()
{
  if (typeof model == 'undefined')
  {
    // Ask for user confirmation
    disableGenerateButton("Downloading the model(200 MB)...");

    // Download the model if it's not cached
    model = await tf.loadLayersModel(modelUrl);

    enableGenerateButton();
  }
}

// Generate a new image
const generateButton = document.getElementById('generateButton');
generateButton.addEventListener('click', generateAndShowImage);

async function generateAndShowImage() 
{
  await loadModel();

  // Disable the generate button
  disableGenerateButton("Generating...");

  // convert canvas to image
  const image = new Image();
  image.src = inputCanvas.toDataURL();
  await new Promise(resolve => image.onload = resolve);

  // Convert to tensor
  const imageTensor = tf.browser.fromPixels(image);

  // Resize
  const resized1 = imageTensor.resizeBilinear([256,256])
  
  // normalize
  const normalized = normalize(resized1);

  // Add batch dimension
  const expandedTensor = tf.expandDims(normalized, axis=0);

  // forward pass
  const prediction = await model.apply(expandedTensor, {'training': true});

  // Remove the batch dimension
  const squeezedTensor  = tf.squeeze(prediction, 0);

  // denormalize the generated image
  const denormalized = denormalize(squeezedTensor);

  // Convert the tensor to pixels
  const imageData = await tf.browser.toPixels(denormalized);

  // Put the generated image onto the canvas
  const imageDataObject = outputCTX.createImageData(outputCanvas.width, outputCanvas.height);
  imageDataObject.data.set(imageData);
  outputCTX.putImageData(imageDataObject, 0, 0);

  // Enable the generate button
  enableGenerateButton();
}


