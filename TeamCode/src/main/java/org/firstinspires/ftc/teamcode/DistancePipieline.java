package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class DistancePipieline extends OpenCvPipeline {

    private static final double KNOWN_HEIGHT_CM = 9.8;
    private static final double FOCAL_LENGTH_PIXELS = 1000; // Calibrate this for your camera

    private double distanceCm = -1;

    public double getDistanceCm() {
        return distanceCm;
    }

    @Override
    public Mat processFrame(Mat input) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

        Scalar lowerBlue = new Scalar(100, 150, 0);
        Scalar upperBlue = new Scalar(140, 255, 255);
        Mat mask = new Mat();
        Core.inRange(hsv, lowerBlue, upperBlue, mask);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        Rect maxRect = null;

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                maxRect = Imgproc.boundingRect(contour);
            }
        }

        if (maxRect != null && maxArea > 500) {
            Imgproc.rectangle(input, maxRect.tl(), maxRect.br(), new Scalar(255, 0, 0), 2);

            double pixelHeight = maxRect.height;
            distanceCm = (KNOWN_HEIGHT_CM * FOCAL_LENGTH_PIXELS) / pixelHeight;

            String label = String.format("Distance: %.1f cm", distanceCm);
            Imgproc.putText(input, label, new Point(maxRect.x, maxRect.y - 10),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
        } else {
            distanceCm = -1;
        }

        return input;
    }
}
