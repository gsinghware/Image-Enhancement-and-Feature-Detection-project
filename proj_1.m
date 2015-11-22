%===================================================
% Computer Vision Programming Assignment 1
% @Zhigang Zhu, 2003-2009
% City College of New York
%===================================================

% ---------------- Step 1 ------------------------
% Read in an image, get information
% type help imread for more information

InputImage = 'IDPicture.bmp'; 
%OutputImage1 = 'IDPicture_bw.bmp';

C1 = imread(InputImage);
[ROWS COLS CHANNELS] = size(C1);

% ---------------- Step 2 ------------------------
% If you want to display the three separate bands
% with the color image in one window, here is 
% what you need to do
% Basically you generate three "color" images
% using the three bands respectively
% and then use [] operator to concatenate the four images
% the orignal color, R band, G band and B band

% First, generate a blank image. Using "uinit8" will 
% give you an image of 8 bits for each pixel in each channel
% Since the Matlab will generate everything as double by default
CR1 =uint8(zeros(ROWS, COLS, CHANNELS));

% Note how to put the Red band of the color image C1 into 
% each band of the three-band grayscale image CR1
for band = 1 : CHANNELS,
    CR1(:,:,band) = (C1(:,:,1));
end

% Do the same thing for G
CG1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CG1(:,:,band) = (C1(:,:,2));
end

% and for B
CB1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CB1(:,:,band) = (C1(:,:,3));
end

% Whenever you use figure, you generate a new figure window 
No1 = figure;  % Figure No. 1

%This is what I mean by concatenation
disimg = [C1, CR1;CG1, CB1]; 

% Then "image" will do the display for you!
image(disimg);

% ---------------- Step 3 ------------------------
% Now we can calculate its intensity image from 
% the color image. Don't forget to use "uint8" to 
% covert the double results to unsigned 8-bit integers

I1    = uint8(round(sum(C1,3)/3));

% You can definitely display the black-white (grayscale)
% image directly without turn it into a three-band thing,
% which is a waste of memeory space

No2 = figure;  % Figure No. 2
image(I1);

% If you just stop your program here, you will see a 
% false color image since the system need a colormap to 
% display a 8-bit image  correctly. 
% The above display uses a default color map
% which is not correct. It is beautiful, though

% ---------------- Step 4 ------------------------
% So we need to generate a color map for the grayscale
% I think Matlab should have a function to do this,
% but I am going to do it myself anyway.

% Colormap is a 256 entry table, each index has three entries 
% indicating the three color components of the index

MAP =zeros(256, 3);

% For a gray scale C[i] = (i, i, i)
% But Matlab use color value from 0 to 1 
% so I scale 0-255 into 0-1 (and note 
% that I do not use "unit8" for MAP

for i = 1 : 256,  % a comma means pause 
    for band = 1:CHANNELS,
        MAP(i,band) = (i-1)/255;
    end 
end

%call colormap to enfore the MAP
colormap(MAP);

% I forgot to mention one thing: the index of Matlab starts from
% 1 instead 0.

% Is it correct this time? Remember the color table is 
% enforced for the current one, which is  the one we 
% just displayed.

% You can test if I am right by try to display the 
% intensity image again:

No3 = figure; % Figure No. 3
image(I1);


% See???
% You can actually check the color map using 
% the edit menu of each figure window

% ---------------- Step 5 ------------------------
% Use imwrite save any image
% check out image formats supported by Matlab
% by typing "help imwrite
% imwrite(I1, OutputImage1, 'BMP');


% ---------------- Step 6 and ... ------------------------
% Students need to do the rest of the jobs from c to g.
% Write code and comments - turn it in both in hard copies and 
% soft copies (electronically)

% ---------------------------------------------------------
% ---------------- PART 1C --------------------------------
% ---------------------------------------------------------

% Generate an intensity image I(x,y) and display it. 
% You should use the equation 
% I = 0.299R + 0.587G + 0.114B (the NTSC standard for luminance).

IntensityImage = (0.299 * CR1) + (0.587 * CG1) + (0.114 * CB1);
No4 = figure;
image(IntensityImage);
title('Intensity Image - I = 0.299R + 0.587G + 0.114B')

% ---------------------------------------------------------
% ---------------- PART 1D --------------------------------
% ---------------------------------------------------------

% Uniformly quantize this image into K levels ( with K=4, 16, 32, 64).
No5 = figure;
K = [4 16 32 64];

for i = 1:length(K)
    I_thres = IntensityImage;
    intervals = round(linspace(1,256,K(i)));
    for j = 1:length(intervals)-1
        I_thres(I_thres > intervals(j) & I_thres < intervals(j+1)) = intervals(j);
    end
    subplot(2, 2, i);
    image(I_thres);
    t = strcat('K = ', num2str(K(i)));    
    title(t);
end


% ---------------------------------------------------------
% ---------------- PART 1E --------------------------------
% ---------------------------------------------------------

No6 = figure;
K = [2 4];

for i = 1:length(K)

    Org_image_a = C1;
    
    Rband = Org_image_a(:, :, 1);
    Gband = Org_image_a(:, :, 2);
    Bband = Org_image_a(:, :, 3);
    
    y = linspace(1, 256, K(i) + 1);
    intervals = round(linspace(0, 256, K(i)))
    
    for j = 1:numel(intervals)-1
        Rband(Rband > y(j) & Rband < y(j+1)) = intervals(j);
        Gband(Gband > y(j) & Gband < y(j+1)) = intervals(j);
        Bband(Bband > y(j) & Bband < y(j+1)) = intervals(j);
    end
    
    Org_image_a = cat(3, Rband, Gband, Bband);
    subplot(1, 2, i);
    image(Org_image_a);
    t = strcat('K = ', num2str(K(i)));    
    title(t);
end

% ---------------------------------------------------------
% ---------------- PART 1F --------------------------------
% ---------------------------------------------------------

Org_image_1 = C1;

Rband_1_1 = Org_image_1(:, :, 1);
Gband_1_1 = Org_image_1(:, :, 2);
Bband_1_1 = Org_image_1(:, :, 3);

c = 0.15;

Rband_1_ = double(Rband_1_1);
Bband_1_ = double(Bband_1_1);
Gband_1_ = double(Gband_1_1);

Rband_1 = c.*log(1+Rband_1_);
Bband_1 = c.*log(1+Bband_1_);
Gband_1 = c.*log(1+Gband_1_);

Org_image_1 = cat(3, Rband_1, Gband_1, Bband_1);

figure

subplot(2, 4, 1);
imshow(C1);
title('Original Image');
subplot(2, 4, 2); 
imshow(Org_image_1);
title('Log Transformation Applied');

subplot(2, 4, 3); 
imshow(Rband_1_1);
title('Red Band Org');
subplot(2, 4, 4); 
imshow(Rband_1);
title('Red Log Band');

subplot(2, 4, 5);
imshow(Gband_1_1);
title('Green Band Org');
subplot(2, 4, 6); 
imshow(Gband_1);
title('Green Log Band');

subplot(2, 4, 7); 
imshow(Bband_1_1);
title('Blue Band Org');
subplot(2, 4, 8); 
imshow(Bband_1);
title('Blue Log Band');


% % ---------------------------------------------------------
% % ---------------- PART 2 --------------------------------
% % ---------------------------------------------------------

% Histogram Equalization
I = imread('IDpicture.bmp');
J = rgb2gray(I);
size(I)
new_img = HistEq(J);
figure, image(I);
figure, image(new_img);
title('Histogram equalization');

% Histogram of the Equalized Image
No8 = figure;
img_hist = Histogram(new_img);
x = linspace(1, 256, 256);
bar(x, img_hist, 'FaceColor', [0 .5 .5], 'EdgeColor', [0 .9 .9], 'LineWidth', 1.5);
set(gca,'XLim',[1 256]);
title('Histogram of the Original Image')

% Thresholding
No9 = figure;
Test_img = IntensityImage;
img_hist = Histogram(Test_img);
[row, col] = size(IntensityImage);
img_size = row * col;
threshold = OtsuThreshold(img_hist, img_size);
Test_img(Test_img > threshold) = 256;
Test_img(Test_img < threshold) = 0;
image(Test_img);
title('Otsu Threshold Applied');

% 
% % ---------------------------------------------------------
% % ---------------- PART 3 --------------------------------
% % ---------------------------------------------------------
% % 

% 1x2 operator

x12 = imread('IDpicture.bmp');
x12_gray = rgb2gray(x12);
x12_gray_d = double(x12_gray);
[x12_row, x12_col] = size(x12_gray_d);
x12_thres = 23;

for i=1:x12_col
    for j=1:x12_row
        if (i == 1 || j == 1 || i == x12_col || j == x12_row)
            x12_image(j, i) = 255;
        else
            sx = (-(x12_gray_d(j, i)) + x12_gray_d(j + 1, i));
            sy = (-(x12_gray_d(j, i)) + x12_gray_d(j, i + 1));
            yM = sqrt(sy.^2);
            xM = sqrt(sx.^2);
            
            M = sqrt(sx.^2 + sy.^2);
            
            x12grad(j, i) = M;
            
            if (M < x12_thres)
                x12_image(j, i) = 0;
            else
                x12_image(j, i) = 255;
            end
            
            if (yM > x12_thres)
                my_y_image_x12(j, i) = 0;
            else
                my_y_image_x12(j, i) = 255;
            end
            
            if (xM > x12_thres)
                my_x_image_x12(j, i) = 0;
            else
                my_x_image_x12(j, i) = 255;
            end
        end
    end
end

figure
subplot(2, 2, 1);
imshow(x12);
title('Original Image')
subplot(2, 2, 2);
imshow(my_y_image_x12);
title('horizontal gradients')
subplot(2, 2, 3); 
imshow(my_x_image_x12);
title('vertical gradients')
subplot(2, 2, 4); 
imshow(x12_image);
title('combined gradients')

figure,imshow(x12grad); 
title('1x2 gradient');


% sobel operator

my_image = imread('IDpicture.bmp');
my_img = rgb2gray(my_image);
grad = my_img;
my_img = double(my_img);
[my_row, my_col] = size(my_img);
my_threshold = 107;

for i=1:my_col
    for j=1:my_row
        
        if (i == 1 || j == 1 || i == my_col || j == my_row)
            my_image_sobel(j, i) = 255;
            my_x_image_sobel(j, i) = 255;
            my_y_image_sobel(j, i) = 255;
        else
            sx = -(my_img(j - 1, i - 1) + 2 * my_img(j, i - 1) + my_img(j + 1, j - 1)) + my_img(j - 1, i + 1) + 2 * my_img(j, i + 1) + my_img(j + 1, i + 1);
            sy = -(my_img(j - 1, i - 1) + 2 * my_img(j - 1, i) + my_img(j - 1, i + 1)) + my_img(j + 1, i - 1) + 2 * my_img(j + 1, i) + my_img(j + 1, i + 1);
            M = sqrt(sx.^2 + sy.^2);
            yM = sqrt(sy.^2);
            xM = sqrt(sx.^2);
            
            grad(j, i) = M;
            
            if (M > my_threshold)
                my_image_sobel(j, i) = 0;
            else
                my_image_sobel(j, i) = 255;
            end
            
            if (yM > my_threshold)
                my_y_image_sobel(j, i) = 0;
            else
                my_y_image_sobel(j, i) = 255;
            end
            
            if (xM > my_threshold)
                my_x_image_sobel(j, i) = 0;
            else
                my_x_image_sobel(j, i) = 255;
            end
        end
    end
end

figure
subplot(2, 2, 1);
imshow(my_image);
title('Original Image')
subplot(2, 2, 2);
imshow(my_y_image_sobel);
title('horizontal gradients')
subplot(2, 2, 3); 
imshow(my_x_image_sobel);
title('vertical gradients')
subplot(2, 2, 4); 
imshow(my_image_sobel);
title('combined gradients')

figure,imshow(grad); 
title('Sobel gradient');


figure;
img_hist = Histogram(grad);
x = linspace(1, 256, 256);
bar(x, img_hist, 'FaceColor', [0 .5 .5], 'EdgeColor', [0 .9 .9], 'LineWidth', 1.5);
set(gca,'XLim',[1 256]);
title('Histogram of the Gradient Image')

cdf = cumsum(my_image_sobel) / sum(my_image_sobel);
index95 = find(cdf >= 0.65, 1, 'first');
grad(grad < index95) = 0;
figure,imshow(grad); 
title('Sobel Edges');
