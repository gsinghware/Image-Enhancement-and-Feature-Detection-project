function hist = Histogram(img)
hist = zeros(1, 256);
for idx = 1:numel(img)
    element = img(idx);
    hist(element+1) = hist(element+1) + 1;
end