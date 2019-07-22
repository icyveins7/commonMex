function effSNR = convertQF2toEffSNR(qf2)
    effSNR = 2.*qf2./(1.0-qf2);
end