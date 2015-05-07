U = []; V = []; U2 = []; V2 = [];
for i = 1:5
    flowim = imread(['000' num2str(i) '_flow.png']);
    [u, v] = img2flow(flowim);
    U = [U; u]; 
    V = [V; v];
    U2 = [U2; u - mean(u(:))];
    V2 = [V2; v - mean(v(:))];
end
dlmwrite('flow_u.txt', U, 'delimiter', ' ', 'precision', '%.2f');
dlmwrite('flow_v.txt', V, 'delimiter', ' ', 'precision', '%.2f');
dlmwrite('flow_u_ms.txt', U2, 'delimiter', ' ', 'precision', '%.2f');
dlmwrite('flow_v_ms.txt', V2, 'delimiter', ' ', 'precision', '%.2f');