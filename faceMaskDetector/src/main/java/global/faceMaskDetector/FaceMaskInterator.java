package global.faceMaskDetector;

import global.faceMaskDetector.GetPropValuesHelper;
import global.faceMaskDetector.FaceMaskInterator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;


import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;


import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class FaceMaskInterator {
    //dataDir is path to folder containing dl4j datasets on local machine
    //parentDir is path to folder containing train and test data sets
    //trainDir and testDir refer to the specific folder containing the train & test data resp.
    private static String dataDir;
    public static String parentDir;
    private static Path trainDir, testDir;
    //Define name of folders containing train and test data
    public static String trainfolder ="train";
    public static String testfolder ="test";
//    public static String testfolder ="validation";


    //Random number to initialise FileSplit when it works on trainData and testData
    private static final int seed = 136;
    private static Random rng = new Random(seed);
    private static FileSplit trainData, testData;
    private static final int nChannels = 3;

    //For kernel. This also sets shape of output CNN layer
    //gridWidth/height and yoloWidth/height should be the same to give square sections
    //must be 32 since the model input is 32 need
    public static final int gridWidth = 4;
    public static final int gridHeight = 4; //should divide yolowidth and yoloheight without remainder?
    //For input image to YOLO?? Does it resize?
    public static final int yolowidth = 128; //next try: 416 400, 384 (384 is max width of training images)
    public static final int yoloheight = 128;

        private static final Logger log = org.slf4j.LoggerFactory.getLogger(FaceMaskInterator.class);

        private static RecordReaderDataSetIterator makeIterator(InputSplit split, Path dir, int batchSize) throws Exception{
            //VOCLabelProvider reads PascalVOC xml label files.
            //Label xml files must have same name as image files.
            //Label xml files must be placed in folder named "Annotations" in same folder as images
            ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(
                    yoloheight,yolowidth,nChannels,gridWidth,
                    gridWidth,new VocLabelProvider(dir.toString()));

            recordReader.initialize(split);
            RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(
                    recordReader,batchSize,1,1,true);
            //Other pre-processors: https://deeplearning4j.org/api/latest/org/nd4j/linalg/dataset/api/preprocessor/package-summary.html
            iter.setPreProcessor(new ImagePreProcessingScaler(0,1));
            return iter;
        }
        public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception{
            return makeIterator(trainData,trainDir,batchSize);
        }
        public static  RecordReaderDataSetIterator testIterator(int batchSize) throws  Exception{
            return makeIterator(testData,testDir,batchSize);
        }

        //setup() and loadData() prepare data to be loaded into the YOLO model
        public static void setup() throws IOException {
            log.info("Loading data......");
            loadData();
            trainDir= Paths.get(parentDir,trainfolder);
            testDir=Paths.get(parentDir,testfolder);

            trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
            testData = new FileSplit(new File(testDir.toString()),NativeImageLoader.ALLOWED_FORMATS,rng);
        }
        private static void loadData() throws IOException,FileNotFoundException {
            // dataDir creates a path of "C:\Users\win10AccountName\.deeplearning4j\data"
            /*
            dataDir= Paths.get(
                    System.getProperty("user.home"),
                    GetPropValuesHelper.getPropValues("dl4j_home.data")
            ).toString();
            parentDir = Paths.get(dataDir,"Facemaskdetection").toString();
            */
            parentDir = new ClassPathResource("dataset").getFile().toString();
            log.info("Folders containing train and test data located \nat: "+parentDir);
        }
    }

