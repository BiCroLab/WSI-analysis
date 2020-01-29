function fv = focus_minimal(I)
% I: image
I = double(I);

% Compute the gradient magnitue,
% gn, has same size as I
% be careful with edge artifacts
sigma = 1;
dx = gpartial(I, 1, sigma); % Partial derivative in first dimension
dy = gpartial(I, 2, sigma); % Partial derivative in second dimension
gn = (dx.^2 + dy.^2).^(1/2);

w = 16; % Window size
% Get a list of mean intensity per [wxw] non-overlapping windows
a = boxmeans(I, w);
b = boxmeans(gn, w);

% Return the proportion of tiles where
% the mean intensity is 12 times the mean gradient
fv = sum(a./b > 12)/numel(a);
end


function m = boxmeans(I, w)

m = zeros(floor(size(I,1)/w)*floor(size(I,2)/w), 1);
idx = 1;
for kk = 1:w:size(I,1)-w
    for ll = 1:w:size(I,2)-w
        P = I(kk:kk+w-1, ll:ll+w-1);
        m(idx) = mean(P(:));
        idx = idx+1;
    end
end

end
