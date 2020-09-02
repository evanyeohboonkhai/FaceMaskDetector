package global.skymind.training.facemaskdetection;

import global.skymind.training.melanomadetector.MelanomaDetector;
import global.skymind.training.melanomadetector.SkinDatasetIterator;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class FaceMaskDetection {
    //***Set model parameters***
    private static final Logger log = LoggerFactory.getLogger(global.skymind.training.melanomadetector.MelanomaDetector.class);
    private static int seed = 123;
    private static double detectionThreshold = 0.5;
    //Set so that nBoxes*(5+nClasses) = odd number
    private static int nBoxes = 5; //refers to anchor boxes to generate at output layer??
    private static double lambdaNoObj = 0.5;
    private static double lambdaCoord = 5.0;
    //Sets aspect ratio of anchor boxes
    private static double[][] priorBoxes = {{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}};

    //***Set model run parameters***
    private static int batchSize = 10; //Smallest batch is lentigoNOS
    private static int nEpochs = 20;
    private static double learningRate = 1e-4;
    //5 types of labelled training data supplied, hence 5 possible outputs:
    //lentigo NOS, lichenoid keratosis, melanoma, nevus, and seborrheic keratosis
    //If this changes, adjust nOut of conv2d_23 at getComputationGraph()
    //This ensures output CNN array dimensions matches that of input at conv2d_1
    private static int nClasses = 3;
    private static List<String> labels;

    //***Set modelFilename and variable for ComputationGraph***
    //Refers to C:\devBox\melanomaDetector\generated-models
    private static File modelFilename = new File(
            System.getProperty("user.dir"),
            "generated-models/melanomaDetector_yolov2.zip");
    private static boolean trigger =true;
    private static ComputationGraph model;


    //***Set bounding boxes parameters***
    private static Frame frame = null;
    //Fix the colour map of the bounding boxes
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar YELLOW = RGB(255, 255, 0);
    private static final Scalar BLUE = RGB(24, 67, 166);


    private static Scalar[] colormap = {GREEN, YELLOW,BLUE};
    //Will later contain labels for bounding boxes
    private static String labeltext = null;



        public static void main(String[] args) throws Exception{
            FaceMaskInterator.setup();
            RecordReaderDataSetIterator trainIter = FaceMaskInterator.trainIterator(batchSize);
            RecordReaderDataSetIterator testIter = FaceMaskInterator.testIterator(1);
            labels = trainIter.getLabels();


            //If model does not exist, train the model, else directly go to model evaluation and then run real time object detection inference.
            //modelFilename.exists()
            if (trigger==false) {
                //STEP 2 : Load trained model from previous execution
                Nd4j.getRandom().setSeed(seed);
                log.info("Load model...");
                model = ModelSerializer.restoreComputationGraph(modelFilename);


            } else {
                Nd4j.getRandom().setSeed(seed);
                INDArray priors = Nd4j.create(priorBoxes);

                //STEP 2 : Train the model using Transfer Learning
                //STEP 2.1: Transfer Learning steps - Load YOLOv2 prebuilt model.
                log.info("Build model...");
                ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();

                //STEP 2.2: Transfer Learning steps - Model Configurations.
                FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

                //STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
                model = getComputationGraph(pretrained, priors, fineTuneConf);
                System.out.println(model.summary(InputType.convolutional(
                        FaceMaskInterator.yoloheight,
                        FaceMaskInterator.yolowidth,
                        nClasses)));

                //STEP 2.4: Training and Save model.
                log.info("Train model...");
                UIServer server = UIServer.getInstance();
                StatsStorage storage = new InMemoryStatsStorage();
                server.attach(storage);
                model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

                for (int i = 1; i < nEpochs + 1; i++) {
                    trainIter.reset();
                    while (trainIter.hasNext()) {
                        model.fit(trainIter.next());
                    }
                    log.info("*** Completed epoch {} ***", i);
                }
                ModelSerializer.writeModel(model, modelFilename, true);
                System.out.println("Model saved.");
            }

            //STEP 3: Evaluate the model's accuracy by using the test iterator.
            OfflineValidationWithTestDataset(testIter);

            //STEP 4: Inference the model and process the webcam stream and make predictions.
            doInference();
        }


        private static ComputationGraph getComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {

            return new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("conv2d_23")
                    .removeVertexKeepConnections("outputs")
                    //The convolution layer just before 'outputs'
                    .addLayer("conv2d_23",
                            new ConvolutionLayer.Builder(1, 1)
                                    .nIn(1024) //no. of input channels
                                    //Setting here determines the dimensions of the final output CNN
                                    .nOut(nBoxes * (5 + nClasses))
                                    .stride(1, 1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "leaky_re_lu_22")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambdaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                    .build(),
                            "conv2d_23")
                    .setOutputs("outputs")
                    .build();
        }
        private static FineTuneConfiguration getFineTuneConfiguration() {

            return new FineTuneConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    .l2(0.00001)
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .build();
        }
        //Visual evaluation of trained object detection model performance
        private static void OfflineValidationWithTestDataset(RecordReaderDataSetIterator test) throws InterruptedException {
            NativeImageLoader imageLoader = new NativeImageLoader();
            CanvasFrame canvas = new CanvasFrame("Validate Test Dataset");
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
            Mat convertedMat = new Mat();
            Mat convertedMat_big = new Mat();



            while (test.hasNext() && canvas.isVisible()) {
                org.nd4j.linalg.dataset.DataSet ds = test.next();
                // get the prediction
                INDArray features = ds.getFeatures();
                INDArray results = model.outputSingle(features);
                //non-maximum suppression (NMS)
                List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
                YoloUtils.nms(objs, 0.5);
                Mat mat = imageLoader.asMat(features);
                mat.convertTo(convertedMat, CV_8U, 255, 0);
                int w = mat.cols() * 2;
                int h = mat.rows() * 2;
                resize(convertedMat, convertedMat_big, new Size(w, h));
                convertedMat_big = drawResults(objs, convertedMat_big, w, h);
                canvas.showImage(converter.convert(convertedMat_big));
                canvas.waitKey();




            }
            canvas.dispose();


        }
        private static Mat drawResults(List<DetectedObject> objects, Mat mat, int w, int h) {
            for (DetectedObject obj : objects) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / FaceMaskInterator.gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / FaceMaskInterator.gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / FaceMaskInterator.gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / FaceMaskInterator.gridHeight);
                //Draw bounding box
                rectangle(mat, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
                //Display label text
                labeltext = label + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
                int[] baseline = {0};
                Size textSize = getTextSize(labeltext, FONT_HERSHEY_TRIPLEX, 0.32, 1, baseline);
                rectangle(mat, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0);
                putText(mat, labeltext, new Point(x1 +1, y2 - 1), FONT_HERSHEY_TRIPLEX,0.32, RGB(0, 0, 0));
            }
            return mat;
        }


        //Stream video frames from Webcam and run them through YOLOv2 model and get predictions
        private static void doInference() {

            String cameraPos = "front";
            int cameraNum = 0;
            Thread thread = null;
            NativeImageLoader loader = new NativeImageLoader(
                    FaceMaskInterator.yolowidth,
                    FaceMaskInterator.yoloheight,
                    3,
                    new ColorConversionTransform(COLOR_BGR2RGB));
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

            if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
                try {
                    throw new Exception("Unknown argument for camera position. Choose between front and back");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            FrameGrabber grabber = null;
            try {
                grabber = FrameGrabber.createDefault(cameraNum);
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

            try {
                grabber.start();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            CanvasFrame canvas = new CanvasFrame("Skin Detection Detection");
            int w = grabber.getImageWidth();
            int h = grabber.getImageHeight();
            canvas.setCanvasSize(w, h);

            while (true) {
                try {
                    frame = grabber.grab();
                } catch (FrameGrabber.Exception e) {
                    e.printStackTrace();
                }

                //if a thread is null, create new thread
                if (thread == null) {
                    thread = new Thread(() ->
                    {
                        while (frame != null) {
                            try {
                                Mat rawImage = new Mat();

                                //Flip the camera if opening front camera
                                if (cameraPos.equals("front")) {
                                    Mat inputImage = converter.convert(frame);
                                    flip(inputImage, rawImage, 1);
                                } else {
                                    rawImage = converter.convert(frame);
                                }

                                Mat resizeImage = new Mat();
                                resize(rawImage, resizeImage, new Size(FaceMaskInterator.yolowidth, FaceMaskInterator.yoloheight));
                                INDArray inputImage = loader.asMatrix(resizeImage);
                                scaler.transform(inputImage);
                                INDArray outputs = model.outputSingle(inputImage);
                                org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
                                List<DetectedObject> objs = yout.getPredictedObjects(outputs, detectionThreshold);
                                YoloUtils.nms(objs, 0.4);
                                rawImage = drawResults(objs, rawImage, w, h);
                                canvas.showImage(converter.convert(rawImage));
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        }
                    });
                    thread.start();
                }

                KeyEvent t = null;
                try {
                    t = canvas.waitKey(33);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                    break;
                }
            }
        }



    }



