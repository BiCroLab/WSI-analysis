% Script to apply watershed on .hdf5 masks from Ilastik - Michele Simonetti
% 17/09/18


%Define a source and a target directory

selpath_source=uigetdir('','Source Directory');
selpath_target=uigetdir('','Target Directory');

%Read all the objects in the source directory and delete the first 3 rows
%of the array

list_files_input=dir([selpath_source '/*.h5']);
% list_files_input=list_files_input(3:end);

%Start the loop through all the .h5 files - first the loop reads the .h5
%files, then it reshape them based on their original format, then it rotate
%the images of 90° left, then it flips vertically the images

for k=1:length(list_files_input)
    %image1 =hdf5info('IDC12879_001_Segmented_mask.h5');
    file_name=strcat([list_files_input(k).folder '/' list_files_input(k).name]);
    image1 = hdf5read(file_name,'exported_data');
    %image1=reshape(image1, [width heigth ]
    image_size=size(image1);
    width=image_size(3);
    height=image_size(2);
    image1 = reshape(image1, [width height]);
    image1 = imrotate(image1,90);
    image1 = flipud(image1);
    image1(image1==1)=1;
    image1(image1==2)=0;
    image1=logical(image1);
%   imshow(image1);
   

%To remove small objects from the image
bw2 = ~bwareaopen(~image1,10);
%imshow(bw2);

%Distance transform
D = -bwdist(~image1);
%imshow(D,[]);

%Run watershed on the masks

Ld = watershed(D);
bw2 = image1;
bw2 (Ld == 0) = 0;
%imshow(bw2)

mask = imextendedmin(D,1);
%imshowpair(image1,mask,'blend')

D2 = imimposemin(D,mask);
Ld2 = watershed(D2);
bw3 = image1;
bw3(Ld2 == 0) = 0;
%check the watershed
%imshow(bw3)

%Save files as HDF5 format
%timage = reshape(uint8(bw3),[1, height, width]);
timage = reshape(uint8(bw3),[1, height, width]);
%h5create([selpath_target,'/',list_files_input(k).name], '/exported_watershed_masks',[1 height width],'Datatype','uint8');
h5create([selpath_target,'/',list_files_input(k).name], '/exported_watershed_masks',[1 height width],'Datatype','uint8');
h5write([selpath_target,'/',list_files_input(k).name], '/exported_watershed_masks', timage);

end        





