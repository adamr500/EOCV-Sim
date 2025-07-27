package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class PositionPipeline extends OpenCvPipeline {

    public enum Position {
        LEFT,
        CENTER,
        RIGHT,
        NOT_FOUND
    }

    private Position position = Position.NOT_FOUND;

    public Position getPosition() {
        return position;
    }

    @Override
    public Mat processFrame(Mat input) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

        Scalar lowerBlue = new Scalar(100, 150, 0);
        Scalar upperBlue = new Scalar(140, 255, 255);
        Mat blueMask = new Mat();
        Core.inRange(hsv, lowerBlue, upperBlue, blueMask);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(blueMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        MatOfPoint largestContour = null;

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }

        if (largestContour != null && maxArea > 500) {
            Rect rect = Imgproc.boundingRect(largestContour);
            Imgproc.rectangle(input, rect.tl(), rect.br(), new Scalar(255, 0, 0), 2);

            int centerX = rect.x + rect.width / 2;
            int width = input.width();

            if (centerX < width / 3) {
                position = Position.LEFT;
            } else if (centerX > 2 * width / 3) {
                position = Position.RIGHT;
            } else {
                position = Position.CENTER;
            }

            Imgproc.putText(input, position.toString(), new Point(rect.x, rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(255, 255, 255), 2);
        } else {
            position = Position.NOT_FOUND;
        }

        return input;
    }
}
