package global.faceMaskDetector;

import com.google.common.reflect.ClassPath;
import global.faceMaskDetector.FaceMaskDetection;
import global.faceMaskDetector.FaceMaskIterator;
import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.*;
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
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.util.*;

import static global.faceMaskDetector.FaceMaskIterator.gridHeight;
import static global.faceMaskDetector.FaceMaskIterator.gridWidth;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.helper.opencv_core.RGB;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;
import static org.nd4j.linalg.util.MathUtils.sigmoid;

public class FaceMaskDetection {
    //***Set model parameters***
    private static final Logger log = LoggerFactory.getLogger(global.faceMaskDetector.FaceMaskDetection.class);
    private static int seed = 123;
    private static double detectionThreshold = 0.12;
    //Set so that nBoxes*(5+nClasses) = odd number
    private static int nBoxes = 6; //=number of priorBoxes
    private static double lambdaNoObj = 0.5;
    private static double lambdaCoord = 5.0;

    /**
     * Bounding box priors dimensions [width, height]. For N bounding boxes, input has shape [rows, columns] = [N,
     * 2] Note that dimensions should be specified as fraction of grid size. For example, a network with 13x13
     * output, a value of 1.0 would correspond to one grid cell; a value of 13 would correspond to the entire
     * image.
     * Current grid: 4X4, for inputs of 720*720. That's a 180*180 grid, for 4px per grid x-y
     * Typical size of masked face in image about 0.5cm*0.5cm to 1cm*1cm
     */
    //{2,2},{1,2},{2,1}, {5,5},{7,5},{5,7}, {10,10},{15,10},{10,15}, {20,20},{20,25},{25,20}, {50,50},{50,55},{55,50}
    //{{4,4},{6,6},{8,8},{10,10},{12,12},{14,14}}
    //{{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}, {5, 10}}
    private static double[][] priorBoxes = {{1, 3}, {1, 5}, {2, 3}, {1, 0.8}, {0.1, 0.5}};

    //***Set model run parameters***
    private static int batchSize = 10;
    private static int nEpochs = 5; //ori: 20. 60 iterations = 1 epoch. 9 max before no improvement
    private static double learningRate = 1e-4;

    //2 output classes: Mask, non mask
    private static int nClasses = 2;
    private static List<String> labels;

    //***Set modelFilename and variable for ComputationGraph***
    //Refers to C:\devBox\FaceMaskDetector\generated-models
    private static File modelFilename = new File(
            System.getProperty("user.dir"),
            "generated-models/faceMaskDetector_yolov2.zip");
    private static boolean trigger =false;
    private static ComputationGraph model;


    //***Set bounding boxes parameters***
    private static Frame frame = null;
    //Fix the colour map of the bounding boxes
    private static final Scalar RED = RGB(255, 0, 0);
    private static final Scalar GREEN = RGB(0, 255, 0);
    //private static final Scalar BLUE = RGB(24, 67, 166);

    private static Scalar[] colormap = {GREEN,RED};
    //Will later contain labels for bounding boxes
    private static String labeltext = null;



    public static void main(String[] args) throws Exception{
        FaceMaskIterator.setup();
        RecordReaderDataSetIterator trainIter = FaceMaskIterator.trainIterator(batchSize);
        RecordReaderDataSetIterator testIter = FaceMaskIterator.testIterator(1);
        labels = trainIter.getLabels();

        //If model does not exist, train the model, else directly go to model evaluation and then run real time object detection inference.
        //modelFilename.exists()
        if (modelFilename.exists()) {
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
                    FaceMaskIterator.yoloheight,
                    FaceMaskIterator.yolowidth,
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
        //videoIdentification();
        //webcamIdentification();
        //imageBoundingBoxDistanceIdentification();
    }


    private static ComputationGraph getComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {

        return new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_23")
                .removeVertexKeepConnections("outputs")

                //The convolution layer just before 'outputs'.
                // This turns the network into a F-CNN, allowing it to accept inputs of varying sizes
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
            //"CV_8U" refers to the matrix type. "8U" means "8 bit" array. "C1" means "single channel". "F" = "floating point"
            mat.convertTo(convertedMat, CV_8UC3, 255, 0);
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
            //Note: If predictions not centred on features, try adjusting gridWidth and gridHeight
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / gridHeight);
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

    private static void videoIdentification() throws InterruptedException, IOException {
        //Sets size of video image that enters the model
        //Multiples of 32 work best. Must follow size that model was trained on??
        //Must be big enough to cover entire input image, yet remain divisible by gridWidth and gridHeight specified
        int vidYolowidth = 256; //width-height ratio of 1.78 in taken images (ori size: 1280*720)
        int vidYoloheight = 256;

        String videoPath = new ClassPathResource("dataset/crowdImages/vid(13).mp4").getFile().toString();
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoPath);
        grabber.setFormat("mp4");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();

        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);

        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();

        canvas.setCanvasSize(w, h);

        NativeImageLoader loader = new NativeImageLoader(vidYolowidth, vidYoloheight, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        System.out.println("Start running video");

        while ((grabber.grab()) != null) {
            Frame frame = grabber.grabImage();
            //if a thread is null, create new thread

            //Apply image transforms
            Mat rawImage = converter.convert(frame);
            Mat resizeImage = new Mat();//rawImage);
            //Mat rotateImage = new Mat();
            resize(rawImage, resizeImage, new Size(vidYolowidth, vidYoloheight));
            //rotate(resizeImage, rotateImage, ROTATE_90_CLOCKWISE);

            INDArray inputImage = loader.asMatrix(resizeImage);
            scaler.transform(inputImage);
            INDArray results = model.outputSingle(inputImage);

            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            YoloUtils.nms(objs, 0.5);

            //Draws bounding boxes
            for (DetectedObject obj : objs) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);

                //Draw bounding box
                rectangle(rawImage, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
                //Display label text
                labeltext = label + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
                int[] baseline = {0};
                Size textSize = getTextSize(labeltext, FONT_HERSHEY_TRIPLEX, 0.32, 1, baseline);
                rectangle(rawImage, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0);
                putText(rawImage, labeltext, new Point(x1 +1, y2 - 1), FONT_HERSHEY_TRIPLEX,0.32, RGB(0, 0, 0));
            }
            canvas.showImage(converter.convert(rawImage));

            KeyEvent t = canvas.waitKey(33);

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
        canvas.dispose();
    }

    //Stream video frames from Webcam and run them through YOLOv2 model and get predictions
    private static void webcamIdentification() {
        String cameraPos = "front";
        int cameraNum = 0;
        Thread thread = null;
        NativeImageLoader loader = new NativeImageLoader(
                FaceMaskIterator.yolowidth,
                FaceMaskIterator.yoloheight,
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
                            resize(rawImage, resizeImage, new Size(FaceMaskIterator.yolowidth, FaceMaskIterator.yoloheight));
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

    //To set ratios of bounding boxes
    private static int[][] anchors = {{10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45}, {59, 119}, {116, 90}, {156, 198}, {373, 326}};
    private static void imageBoundingBoxDistanceIdentification() throws IOException {
        int safeDistance = 80;
        model.init();
        System.out.println(model.summary());

        String testImagePATH = new ClassPathResource("dataset/test/img(3).jpg").getFile().toString();
        Mat opencvMat = imread(testImagePATH);
        NativeImageLoader nil = new NativeImageLoader(gridWidth, gridHeight, 3);
        INDArray input = nil.asMatrix(testImagePATH).div(255);

        //DL4j defaults to NCHW, need to convert to NHWC(channel last). This is a problem with the Keras model...
        //Doing 'permute' changes the dimensions of the array.
        //Ref: https://www.mathworks.com/help/matlab/ref/permute.html
        //input = input.permute(0, 2, 3, 1);

        //Should change this 'input' to supply images as Mat, instead of NDArray like this example (probably cuz Keras model)
        List<DetectedObject> objs = getPredictedObjects(input);
        YoloUtils.nms(objs, 0.4);

        int w = opencvMat.cols();
        int h = opencvMat.rows();

        //TODO:Use set instead of arraylist to ensure no duplicates!
        List<INDArray> centers = new ArrayList<>();
        List<INDArray> people = new ArrayList<>();
        Set violators = new HashSet<INDArray>();

        int centerX;
        int centerY;

        for (DetectedObject obj : objs) {
            //0 is the index of "person" in COCO dataset list of objects
            if (obj.getPredictedClass() == 0) {
                //            Scale the coordinates back to full size
                centerX = (int) obj.getCenterX() * w / gridWidth;
                centerY = (int) obj.getCenterY() * h / gridHeight;

                circle(opencvMat, new Point(centerX, centerY), 2, new Scalar(0, 255, 0, 0), 2, 0, 0);
                //            Draw bounding boxes on opencv mat
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                //            Scale the coordinates back to full size
                xy1[0] = xy1[0] * w / gridWidth;
                xy1[1] = xy1[1] * h / gridHeight;
                xy2[0] = xy2[0] * w / gridWidth;
                xy2[1] = xy2[1] * h / gridHeight;

                //Draw bounding box
                rectangle(opencvMat, new Point((int) xy1[0], (int) xy1[1]), new Point((int) xy2[0], (int) xy2[1]), new Scalar(0, 255, 0, 0), 2, LINE_8, 0);
                centers.add(Nd4j.create(new float[]{(float) centerX, (float) centerY}));
                people.add(Nd4j.create(new float[]{(float) xy1[0], (float) xy1[1], (float) xy2[0], (float) xy2[1]}));
            }
        }

        //Calculate the euclidean distance between all pairs of center points
        for (int i = 0; i < centers.size(); i++) {
            for (int j = 0; j < centers.size(); j++) {
                double distance = euclideanDistance(centers.get(i), centers.get(j));
                if (distance < safeDistance && distance > 0) {
                    line(opencvMat, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)),
                            new Point(centers.get(j).getInt(0), centers.get(j).getInt(1)), Scalar.RED, 2, 1, 0);

                    violators.add(centers.get(i));
                    violators.add(centers.get(j));

                    int xmin = people.get(i).getInt(0);
                    int ymin = people.get(i).getInt(1);
                    int xmax = people.get(i).getInt(2);
                    int ymax = people.get(i).getInt(3);

                    rectangle(opencvMat, new Point(xmin, ymin), new Point(xmax, ymax), Scalar.RED, 2, LINE_8, 0);
                    circle(opencvMat, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)), 3, Scalar.RED, -1, 0, 0);
                }
            }
        }
        putText(opencvMat, String.format("Number of people: %d", people.size()), new Point(10, 30), 4, 1.0, new Scalar(0, 255, 0, 0), 2, LINE_8, false);
        putText(opencvMat, String.format("Number of violators: %d", violators.size()), new Point(10, 60), 4, 1.0, new Scalar(0, 0, 255, 0), 2, LINE_8, false);
        imshow("Social Distancing Monitor", opencvMat);

        if (waitKey(0) == 27) {
            destroyAllWindows();
        }

    }
    private static List<DetectedObject> getPredictedObjects(INDArray input) {
        INDArray[] outputs = model.output(input);

        List<DetectedObject> out = new ArrayList();
        float detectionThreshold = 0.6f;
        // Each cell had information for 3 boxes
        // box info starts from indices {0,85,170}
        int[] boxOffsets = {0, nClasses + 5, (nClasses + 5) * 2};
        int exampleNum_in_batch = 0;


        for (int layerNum = 0; layerNum < 3; layerNum++) {
            long cellGridWidth = outputs[layerNum].shape()[1];
            long cellGridHeight = outputs[layerNum].shape()[2];
            float cellWidth = gridWidth / cellGridWidth;
            float cellHeight = gridHeight / cellGridHeight;

            for (int i = 0; i < cellGridHeight; i++) {
                for (int j = 0; j < cellGridWidth; j++) {
                    float centerX;
                    float centerY;
                    float width;
                    float height;
                    int anchorInd;

                    for (int k = 0; k < 3; k++) {
//                        exampleNum_in_batch is 0 because there is only 1 example in the batch
//                        getFloat(),get() has 4 arguments because there are 4 indices we can use to get the single float value we want, in the order NHWC
                        float prob = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 4});
                        if (prob > detectionThreshold) {
//                            TODO: class probabilities does not make sense
                            INDArray classes_scores = outputs[layerNum].get(
                                    point(exampleNum_in_batch),
                                    point(i),
                                    point(j),
                                    NDArrayIndex.interval(boxOffsets[k] + 5, boxOffsets[k] + nClasses + 5));

                            centerX = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 0});
                            centerY = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 1});
                            width = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 2});
                            height = outputs[layerNum].getFloat(new int[]{exampleNum_in_batch, i, j, boxOffsets[k] + 3});

                            anchorInd = (2 - layerNum) * 3 + k;

                            centerX = (float) ((sigmoid(centerX) + j) * cellWidth);
                            centerY = (float) ((sigmoid(centerY) + i) * cellHeight);

                            width = (float) (Math.exp(width)) * anchors[anchorInd][0];
                            height = (float) (Math.exp(height)) * anchors[anchorInd][1];

                            out.add(new DetectedObject(k, centerX, centerY, width, height, classes_scores, prob));
                        }
                    }
                }
            }
        }
        return out;
    }
}



