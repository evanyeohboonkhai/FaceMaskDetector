package global.faceMaskDetector;

import global.faceMaskDetector.FaceMaskDetection;
import global.faceMaskDetector.FaceMaskIterator;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.awt.event.KeyEvent;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

//public class VideoIdentification {
//    private static ComputationGraph model = FaceMaskDetection.model;
//    private static Frame frame = null;
//    private static
//
//    //Stream video frames from Webcam and run them through YOLOv2 model and get predictions
//    private static void videoIdentification() {
//        String cameraPos = "front";
//        int cameraNum = 0;
//        Thread thread = null;
//        NativeImageLoader loader = new NativeImageLoader(
//                FaceMaskIterator.yolowidth,
//                FaceMaskIterator.yoloheight,
//                3,
//                new ColorConversionTransform(COLOR_BGR2RGB));
//        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
//
//        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
//            try {
//                throw new Exception("Unknown argument for camera position. Choose between front and back");
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//        }
//
//        FrameGrabber grabber = null;
//        try {
//            grabber = FrameGrabber.createDefault(cameraNum);
//        } catch (FrameGrabber.Exception e) {
//            e.printStackTrace();
//        }
//        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
//
//        try {
//            grabber.start();
//        } catch (FrameGrabber.Exception e) {
//            e.printStackTrace();
//        }
//
//        CanvasFrame canvas = new CanvasFrame("Skin Detection Detection");
//        int w = grabber.getImageWidth();
//        int h = grabber.getImageHeight();
//        canvas.setCanvasSize(w, h);
//
//        while (true) {
//            try {
//                frame = grabber.grab();
//            } catch (FrameGrabber.Exception e) {
//                e.printStackTrace();
//            }
//
//            //if a thread is null, create new thread
//            if (thread == null) {
//                thread = new Thread(() ->
//                {
//                    while (frame != null) {
//                        try {
//                            Mat rawImage = new Mat();
//
//                            //Flip the camera if opening front camera
//                            if (cameraPos.equals("front")) {
//                                Mat inputImage = converter.convert(frame);
//                                flip(inputImage, rawImage, 1);
//                            } else {
//                                rawImage = converter.convert(frame);
//                            }
//
//                            Mat resizeImage = new Mat();
//                            resize(rawImage, resizeImage, new Size(FaceMaskIterator.yolowidth, FaceMaskIterator.yoloheight));
//                            INDArray inputImage = loader.asMatrix(resizeImage);
//                            scaler.transform(inputImage);
//                            INDArray outputs = model.outputSingle(inputImage);
//                            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
//                            List<DetectedObject> objs = yout.getPredictedObjects(outputs, detectionThreshold);
//                            YoloUtils.nms(objs, 0.4);
//                            rawImage = drawResults(objs, rawImage, w, h);
//                            canvas.showImage(converter.convert(rawImage));
//                        } catch (Exception e) {
//                            throw new RuntimeException(e);
//                        }
//                    }
//                });
//                thread.start();
//            }
//
//            KeyEvent t = null;
//            try {
//                t = canvas.waitKey(33);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//
//            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
//                break;
//            }
//        }
//    }
//}
