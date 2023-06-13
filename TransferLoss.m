classdef TransferLoss <handle
    % Member Variables
    properties
      % Constructor for TransferLoss class
      loss_type="cosine";
      input_dim=512;
    end
    % Member Functions
    methods
        function obj=TransferLoss(loss_type, input_dim)
               obj.loss_type = loss_type;
               obj.input_dim = input_dim;
        end
        
        function loss=compute(obj, X, Y)
            switch obj.loss_type
%                 case "mmd_lin" || "mmd"
%                     mmdloss = mmd.MMD_loss(kernel_type='linear');
%                     loss = mmdloss(X, Y);
                case "coral"
                    loss = CORAL(X, Y);
                case "adv"
                    loss = adv(X, Y);
                case "cosine" ||  "cos"
                    loss = 1 - cosine(X, Y);
                case "kl"
                    loss = kl_js.kl_div(X, Y);
                case "js"
                    loss = kl_js.js(X, Y);
                case "mine"
                    mine_model = mutual_info.Mine_estimator(input_dim=self.input_dim, hidden_dim=60).cuda();
                    loss = mine_model(X, Y);

            end
        end   
    end 
end