function img1 = HistEq(img)

% first step
% count the total number of pixels associated with each pixel intensity
% frequency = Histogram(img)
frequency = zeros(1, 256);
for idx = 1:numel(img)
    element = img(idx);
    frequency(element+1) = frequency(element+1) + 1;
end

% second step
% calculate the prob of each pixel intensity in the image matrix
[row, col] = size(img);
img_size = row * col;

img1 = uint8(zeros(row, col));

probability = frequency/img_size;

% cumulative probability
cum_prob = cumsum(probability);

% CP * new range
range = 256;
new_CP = floor(cum_prob * range);

for i=1:row
    for j=1:col
        img1(i, j) = new_CP(img(i, j) + 1);
    end
end
