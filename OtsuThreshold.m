% Otsu's Thresholding Algorithm
% Reference Video - Otsu-s Segmentation with Demo - Duration 1425
% https://www.youtube.com/watch?v=b4Hg2t_tmn4
% By: Gurpreet Singh

function threshold = OtsuThreshold(hist, size)
% minimize within-class variance
min_within_class_variance = 0;
min_within_class_variance_idx = 0;

for i=1:256
    % CLASS 1 in bi-modal histogram (Background)
    
    % sum of the frequencies
    X = hist(1:i);
    qt1 = sum(X);
    qtz = qt1/size;
    
    % mean class variance
    mean_t1 = 0;
    variance_t1 = 0;
    
    if (qt1 > 0)
        for j=1:i
            mean_t1 = mean_t1 + (j * hist(j));
        end
        
        mean_t1 = (mean_t1 / qt1);
        
        for j=1:i
            variance_t1 = variance_t1 + (((j - mean_t1).^2) * hist(j));
        end
        variance_t1 = (variance_t1 / qt1);
    end
    
    % CLASS 2 in bi-modal histogram (Foreground)
    X = hist(i:256);
    qt2 = sum(X);
    qtl = qt2/size;
    
    mean_t2 = 0;
	variance_t2 = 0;
       
    if (qt2 > 0)
        for j=i:256
            mean_t2 = mean_t2 + (j * hist(j));
        end
        mean_t2 = (mean_t2 / qt2);
        
        for j=i:256
            variance_t2 = variance_t2 + (((j - mean_t2).^2) * hist(j));
        end
        variance_t2 = (variance_t2 / qt2);
    end
    
    variance_w_t_sq = qtz * variance_t1 + qtl * variance_t2;
    
    if (i == 1)
        min_within_class_variance = variance_w_t_sq;
        min_within_class_variance_idx = i;
    end
    
    if (variance_w_t_sq < min_within_class_variance)
        min_within_class_variance = variance_w_t_sq;
        min_within_class_variance_idx = i;
    end
    
end


threshold = min_within_class_variance_idx;