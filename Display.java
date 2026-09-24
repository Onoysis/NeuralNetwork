import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

public class Display extends JPanel {

    private final BufferedImage image;

    private int lastX;
    private int lastY;

    private boolean drawing = false;
    private boolean erasing = false;

    private static final int WIDTH = 280;
    private static final int HEIGHT = 280;

    // Increase/decrease this to change pen thickness.
    private static final int BRUSH_SIZE = 20;

    public Display() {
        image = new BufferedImage(
                WIDTH,
                HEIGHT,
                BufferedImage.TYPE_INT_RGB
        );

        clear();

        setPreferredSize(new Dimension(WIDTH, HEIGHT));

        MouseAdapter mouseHandler = new MouseAdapter() {

            @Override
            public void mousePressed(MouseEvent e) {
                drawing = true;

                lastX = e.getX();
                lastY = e.getY();

                // Also draws a dot if the user only clicks.
                drawLine(lastX, lastY, lastX, lastY);
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                if (!drawing) {
                    return;
                }

                int x = e.getX();
                int y = e.getY();

                drawLine(lastX, lastY, x, y);

                lastX = x;
                lastY = y;
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                drawing = false;
            }
        };

        addMouseListener(mouseHandler);
        addMouseMotionListener(mouseHandler);
    }

    private void drawLine(int x1, int y1, int x2, int y2) {
        Graphics2D g = image.createGraphics();

        g.setColor(erasing ? Color.BLACK : Color.WHITE);

        g.setStroke(new BasicStroke(
                BRUSH_SIZE,
                BasicStroke.CAP_ROUND,
                BasicStroke.JOIN_ROUND
        ));

        g.drawLine(x1, y1, x2, y2);
        g.dispose();

        repaint();
    }

    public void clear() {
        Graphics2D g = image.createGraphics();

        g.setColor(Color.BLACK);
        g.fillRect(0, 0, image.getWidth(), image.getHeight());

        g.dispose();

        repaint();
    }

    public void setErasing(boolean erasing) {
        this.erasing = erasing;
    }

    /*
     * Returns every pixel as a value between 0.0 and 1.0.
     *
     * 0.0 = black
     * 1.0 = white
     *
     * Pixels are stored row by row:
     *
     * (0,0), (1,0), (2,0), ...
     * (0,1), (1,1), (2,1), ...
     */
    public double[] getPixels() {
        int size = 28;
    
        BufferedImage resized = new BufferedImage(
                size,
                size,
                BufferedImage.TYPE_INT_RGB
        );
    
        Graphics2D g = resized.createGraphics();
    
        g.drawImage(
                image,
                0, 0,
                size, size,
                null
        );
    
        g.dispose();
    
        double[] pixels = new double[size * size];
    
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
    
                int rgb = resized.getRGB(x, y);
    
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;
    
                double brightness =
                        (red + green + blue) / (3.0 * 255.0);
    
                pixels[y * size + x] = brightness;
            }
        }
    
        return pixels;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        g.drawImage(image, 0, 0, null);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {

            JFrame frame = new JFrame("Digit Drawer");

            Display drawer = new Display();

            JButton drawButton = new JButton("Draw");
            JButton eraseButton = new JButton("Erase");
            JButton clearButton = new JButton("Clear");
            JButton recognizeButton = new JButton("Recognize");

            drawButton.addActionListener(e -> {
                drawer.setErasing(false);
            });

            eraseButton.addActionListener(e -> {
                drawer.setErasing(true);
            });

            clearButton.addActionListener(e -> {
                drawer.clear();
            });

            recognizeButton.addActionListener(e -> {
                double[] pixels = drawer.getPixels();

                System.out.println(
                        "Number of pixels: " + pixels.length
                );

                /*
                 * Eventually:
                 *
                 * double[] output = neuralNetwork.predict(pixels);
                 *
                 * int predictedNumber = ...
                 */
            });

            JPanel buttons = new JPanel();

            buttons.add(drawButton);
            buttons.add(eraseButton);
            buttons.add(clearButton);
            buttons.add(recognizeButton);

            frame.setLayout(new BorderLayout());

            frame.add(drawer, BorderLayout.CENTER);
            frame.add(buttons, BorderLayout.SOUTH);

            frame.pack();

            frame.setDefaultCloseOperation(
                    JFrame.EXIT_ON_CLOSE
            );

            frame.setResizable(false);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
