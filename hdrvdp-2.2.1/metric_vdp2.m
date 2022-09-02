function [vdp2] = metric_vdp2(img, ref)
    sizes = [size(img, 1), size(img, 2)];
    % img(img > 1) = 1;
    % ref(ref > 1) = 1;
    img = img .* 4000.0;
    ref = ref .* 4000.0;
    ppd = hdrvdp_pix_per_deg( 21, sizes, 1 );
    res1 = hdrvdp( img, ref, 'rgb-bt.709', ppd );
    vdp2 = res1.Q;
end
