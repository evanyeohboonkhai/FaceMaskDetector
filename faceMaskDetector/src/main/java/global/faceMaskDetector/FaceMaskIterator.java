package global.faceMaskDetector;

import global.faceMaskDetector.GetPropValuesHelper;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;


import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2BGR;

public class FaceMaskIterator {
    //dataDir is path to folder containing dl4j datasets on local machine
    //parentDir is path to folder containing train and test data sets
    //trainDir and testDir refer to the specific folder containing the train & test data resp.
    private static String dataDir;
    public static String parentDir;
    private static Path trainDir, testDir;
    //Define name of folders containing train and test data
    public static String trainfolder ="trainResized";
    public static String testfolder ="test(2)Resized";
//    public static String testfolder ="validation";


    //Random number to initialise FileSplit when it works on trainData and testData
    private static final int seed = 136;
    private static Random rng = new Random(seed);
    private static FileSplit trainData, testData;
    private static final int nChannels = 3;

    //For kernel. This also sets shape of output CNN layer
    //gridWidth/height and yoloWidth/height should be the same to give square sections
    //gridWidth/Height *32 = yolodWidth/height
    //Default: 416*416, for 24*24 grid
    public static final int gridWidth = 8;
    public static final int gridHeight = 8;
    //For setting size of input image to YOLO model. Does it resize?
    public static final int yolowidth = 256; //next try: 416 400, 384 (384 is max width of training images)
    public static final int yoloheight = 256;

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FaceMaskIterator.class);

    private static RecordReaderDataSetIterator makeTrainIterator(
            InputSplit split, Path dir, int batchSize)
    throws Exception{
        //Only the "train" set should apply transforms
        //Use this to produce random transformations
        Random rng = new Random(seed);

        //Wait... will these transforms mess with the bounding boxes I drew?!! After all, the boxes have fixed coordinates
        //Specify some image transforms
        //Flips/mirrors image horizontally
        ImageTransform flipHorizontal = new FlipImageTransform(0);
        //Sets maximum crop in top, left, bottom, and right to 700 px
        ImageTransform randomCrop = new CropImageTransform(rng, 150);
        //Sets maxDeviationInX, maxDeviationInY, maxRotation, maxScaling relative to 1
        ImageTransform rotate = new RotateImageTransform(rng, 70);
        //Sets maximum change in x and y when scaling
        //Cannot use Random here, it will cause "error: (-215:Assertion failed) inv_scale_x > 0 in function 'cv::resize'"
        ImageTransform scale = new ScaleImageTransform(150);
        //Changes img colour to account for different image colour channels
        //Conversion codes: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
        ImageTransform colorShift = new ColorConversionTransform(rng, COLOR_BGRA2BGR);
        //Accounts for change in perspective
        ImageTransform warp = new WarpImageTransform(rng, 300);

        //Gather the transforms into a List<>
        //List accepts 2 variables in each entry:
        //The ImageTransform to be applied and a Double to specify odds that the ImageTransform will be applied
        List<Pair<ImageTransform, Double>> transformPipeline = Arrays.asList(
            new Pair<>(flipHorizontal, 0.6),
            new Pair<>(randomCrop, 0.6),
            new Pair<>(rotate, 0.6),
            new Pair<>(scale, 0.6)
            //new Pair<>(colorShift, 0.7),
            //new Pair<>(warp, 0.7) //might be screwing with model...
        );
        //Set up an ImageTransform-er to apply the transformPipeline defined above
        //Setting shuffle=true randomly shuffles the pipeline with each transform,
        //further increasing the available dataset.
        ImageTransform transform = new PipelineImageTransform(transformPipeline, true);

        //VOCLabelProvider reads PascalVOC xml label files.
        //Label xml files must have same name as image files.
        //Label xml files must be placed in folder named "Annotations" in same folder as images
        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(
            yoloheight,yolowidth,nChannels,gridHeight,
            gridWidth,new VocLabelProvider(dir.toString())
        );
        recordReader.initialize(split);

        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(
                recordReader,batchSize,1,1,true);
        //Other pre-processors: https://deeplearning4j.org/api/latest/org/nd4j/linalg/dataset/api/preprocessor/package-summary.html
        iter.setPreProcessor(new ImagePreProcessingScaler(0,1));
        return iter;
    }
    private static RecordReaderDataSetIterator makeTestIterator(
            InputSplit split, Path dir, int batchSize)
    throws Exception{
        //VOCLabelProvider reads PascalVOC xml label files.
        //Label xml files must have same name as image files.
        //Label xml files must be placed in folder named "Annotations" in same folder as images
        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(
            yoloheight,yolowidth,nChannels,gridHeight,
            gridWidth,new VocLabelProvider(dir.toString())
        );
        recordReader.initialize(split);

        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(
                recordReader,batchSize,1,1,true);
        //Other pre-processors: https://deeplearning4j.org/api/latest/org/nd4j/linalg/dataset/api/preprocessor/package-summary.html
        iter.setPreProcessor(new ImagePreProcessingScaler(0,1));
        return iter;
    }
    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception{
        return makeTrainIterator(trainData,trainDir,batchSize);
    }
    public static  RecordReaderDataSetIterator testIterator(int batchSize) throws  Exception{
        return makeTestIterator(testData,testDir,batchSize);
    }

    //setup() and loadData() prepare data to be loaded into the YOLO model
    public static void setup() throws IOException {
        log.info("Loading data......");
        loadData();
        trainDir= Paths.get(parentDir,trainfolder);
        testDir=Paths.get(parentDir,testfolder);

        log.info("Train data located at: "+trainDir);
        log.info("Test data located at: "+testDir);

        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()),NativeImageLoader.ALLOWED_FORMATS,rng);
    }
    private static void loadData() throws IOException,FileNotFoundException {
        // dataDir creates a path of "C:\Users\win10AccountName\.deeplearning4j\data"
        dataDir= Paths.get(
                System.getProperty("user.home"),
                GetPropValuesHelper.getPropValues("dl4j_home.data")
        ).toString();
        parentDir = Paths.get(dataDir,"faceMaskDetector").toString();

        //Use if data in /java/resources folder in repo
        //parentDir = new ClassPathResource("dataset").getFile().toString();
        log.info("Folders containing train and test data located \nat: "+parentDir);
    }
}

