package org.firstinspires.ftc.teamcode;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import org.openftc.easyopencv.OpenCvPipeline;
import org.openftc.easyopencv.OpenCvCamera;

public class FirstPipeline extends OpenCvPipeline {

    private int frameCount = 0;

    @Override
    public Mat processFrame(Mat input) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

        // Define color ranges
        Scalar lowerRed1 = new Scalar(0, 100, 100);
        Scalar upperRed1 = new Scalar(10, 255, 255);
        Scalar lowerRed2 = new Scalar(160, 100, 100);
        Scalar upperRed2 = new Scalar(179, 255, 255);
        Scalar lowerYellow = new Scalar(20, 100, 100);
        Scalar upperYellow = new Scalar(35, 255, 255);
        Scalar lowerBlue = new Scalar(100, 150, 0);
        Scalar upperBlue = new Scalar(140, 255, 255);

        // Red mask (two ranges)
        Mat redMask1 = new Mat();
        Mat redMask2 = new Mat();
        Core.inRange(hsv, lowerRed1, upperRed1, redMask1);
        Core.inRange(hsv, lowerRed2, upperRed2, redMask2);
        Mat redMask = new Mat();
        Core.addWeighted(redMask1, 1.0, redMask2, 1.0, 0.0, redMask);

        // Yellow and blue masks
        Mat yellowMask = new Mat();
        Core.inRange(hsv, lowerYellow, upperYellow, yellowMask);
        Mat blueMask = new Mat();
        Core.inRange(hsv, lowerBlue, upperBlue, blueMask);

        // Draw boxes
        drawLargestBlob(redMask, input, new Scalar(0, 0, 255), "Red");
        drawLargestBlob(yellowMask, input, new Scalar(0, 255, 255), "Yellow");
        drawLargestBlob(blueMask, input, new Scalar(255, 0, 0), "Blue");

        return input;
    }

    // Helper function
    private void drawLargestBlob(Mat mask, Mat input, Scalar color, String labelPrefix) {
        double nonZero = Core.countNonZero(mask);
        double total = mask.rows() * mask.cols();
        double probability = (nonZero / total) * 100.0;

        java.util.List<org.opencv.core.MatOfPoint> contours = new java.util.ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        org.opencv.core.MatOfPoint largestContour = null;
        for (org.opencv.core.MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }

        if (largestContour != null && maxArea > 500) {
            Rect rect = Imgproc.boundingRect(largestContour);
            Imgproc.rectangle(input, rect.tl(), rect.br(), color, 3);
            String label = String.format("%s: %.1f%%", labelPrefix, probability);
            Imgproc.putText(input, label, new Point(rect.x, rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
    }

    @Override
    public void onViewportTapped() {
        // Optional: handle UI tap in sim
    }
}