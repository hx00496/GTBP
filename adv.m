function loss = adv(source,target)
% Adversarial Network Discriminator
layerDiscriminator=[
    sequenceInputLayer(64,"Name","input")
    fullyConnectedLayer(32,"Name","fc_1")
    fullyConnectedLayer(1,"Name","fc_2")
];
Discriminator = layerGraph(layerDiscriminator);
dlDiscriminator= dlnetwork(Discriminator);
% Calculate Loss
domain_src=ones(1,size(source,2));
domain_tar=zeros(1,size(target,2));
source=dlarray(source.squeeze,"CT");
target=dlarray(target.squeeze,"CT");
pred_src =forward(dlDiscriminator,source);
pred_tar= forward(dlDiscriminator,target);
loss_src = crossentropy(pred_src,domain_src);
loss_tar = crossentropy(pred_tar,domain_tar);
loss= loss_src+loss_tar;
end


