package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class ContourPipeline extends OpenCvPipeline {

    private double angle = 0;

    @Override
    public Mat processFrame(Mat input) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

        // Define blue range
        Scalar lowerBlue = new Scalar(100, 150, 50);
        Scalar upperBlue = new Scalar(140, 255, 255);
        Mat mask = new Mat();
        Core.inRange(hsv, lowerBlue, upperBlue, mask);

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Point pt1 = null;
        Point pt2 = null;
        double maxLength = 0;

        for (MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, 5, true);
            Point[] points = approx.toArray();

            for (int i = 0; i < points.length; i++) {
                Point p1 = points[i];
                Point p2 = points[(i + 1) % points.length];
                double length = Math.hypot(p2.x - p1.x, p2.y - p1.y);
                if (length > maxLength) {
                    maxLength = length;
                    pt1 = p1;
                    pt2 = p2;
                }
            }
        }

        if (pt1 != null && pt2 != null) {
            Imgproc.line(input, pt1, pt2, new Scalar(255, 0, 0), 2);
            double dx = pt2.x - pt1.x;
            double dy = pt2.y - pt1.y;

            // Always compute angle left-to-right
            if (dx < 0) {
                dx = -dx;
                dy = -dy;
            }

            angle = Math.toDegrees(Math.atan2(dy, dx));
            if (angle < 0) {
                angle += 360;
            }
            Imgproc.putText(input, String.format("Angle: %.2f", angle),
                    pt1, Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
        }

        return input;
    }

    public double getAngle() {
        return angle;
    }
}
