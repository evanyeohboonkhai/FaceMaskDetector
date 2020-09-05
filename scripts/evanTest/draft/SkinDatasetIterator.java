package global.evanTest;

import global.skymind.Helper;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.shade.protobuf.compiler.PluginProtos;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class SkinDatasetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(SkinDatasetIterator .class);
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
    private static String downloadLink;
    private static Path trainDir, testDir;
    private static FileSplit trainData, testData;
    private static final int nChannels = 3;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;
    public static final int yolowidth = 416;
    public static final int yoloheight = 416;
    //Define you own folder name
    public static String foldername  ="cell";
    public static String trainfolder ="train";
    public static String testfolder ="train";

    private static RecordReaderDataSetIterator makeIterator(InputSplit split,Path dir, int batchSize) throws Exception{

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloheight,yolowidth,nChannels,gridWidth
                ,gridWidth,new VocLabelProvider(dir.toString()));

        recordReader.initialize(split);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader,batchSize,1,1,true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0,1));
        return iter;

    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception{
        return makeIterator(trainData,trainDir,batchSize);
    }

    public static  RecordReaderDataSetIterator testIterator(int batchSize) throws  Exception{
        return makeIterator(testData,testDir,batchSize);

    }

    public static void setup() throws IOException{
        log.info("Loading data......");
        loadData();
        trainDir=Paths.get(dataDir,foldername,trainfolder);
        testDir=Paths.get(dataDir,foldername,testfolder);
        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()),NativeImageLoader.ALLOWED_FORMATS,rng);



    }
    private static void loadData() throws IOException{
        dataDir= Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();
        File parentDir = new File(Paths.get(dataDir,foldername).toString());
        )
    }


}
}