function blob()

% Assumes that files starts with 'i', this was the simplest way to
% exclude files that start with '.'
files = dir('i*.tif');

outFile = fopen('blob.csv', 'w');
for ff = 1:numel(files)
    fname = files(ff).name;
    I = imread(fname);
    I = double(I);
    I = I/(2^16-1);    
    [focus, blob] = fvalue(I);
    fprintf(outFile, "%s, %f, %f\n", fname, focus, blob);
    fprintf("%s, %f, %f\n", fname, focus, blob);
end
fclose(outFile);

end

function [f, b] = fvalue(I)
% Return a "focus" value f, and a "blob" value b
    sigma = 1;
    dx = gpartial(I, 1, sigma);
    dy = gpartial(I, 2, sigma);
    gn = (dx.^2 + dy.^2).^(1.2);
    crop = 10;
    gn = gn(crop+1:end-crop, crop+1:end-crop);
    f = max(gn(:));
    side = 50;
    b = 0;
    for kk = 1:side/2:size(I,1)
        for ll = 1:side/2:size(I,2)
            if(kk+side < size(I,1))
                if(ll+side < size(I,2))
                    P = I(kk:kk+side, ll:ll+side);
                    b = max(b, mean(P(:)));
                end
            end
        end
    end
                    
                
end
