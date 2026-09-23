% Deal with relative paths for the PDF and XML files.
% This assumes that the paths are the same as in the pdf2ecg Github repo
% If you change the file/folder structure, this may not work and you
% can just replace pdfFile and xmlFile with the actual paths
here = fileparts(mfilename('fullpath'));
repo = fileparts(here);
addpath(here);

% Set paths for PDF and XML files
pdfFile = fullfile(here, 'ecgs', 'ecg_12x1.pdf');
xmlFile = fullfile(here, 'ecgs', 'ecg.xml');

% pdfFile = '<path to PDF File>';
% xmlFile = '<path to XML File>';


% PDF data
[signals, info] = load_musepdf(pdfFile,'verbose');
II_pdf = signals.II(:);     % Force column vector

% XML data
[hz, I, II, V1, V2, V3, V4, V5, V6] = load_musexml(xmlFile);
II_xml = II(:);    % Force column vector

% Show that XML and PDF leads are exactly equal
% Be alert for floating point errors as we are comparing mV
exact = isequal(II_pdf,II_xml);
fprintf('PDF and XML leads are equal: %s\n', string(exact));

% Plot XML and PDF leads
figure
hold on
title('Comparison of XML and Reconstructed PDF Waveforms')
plot(II_xml, 'k', 'LineWidth', 2 ,'DisplayName','XML')
plot(II_pdf, 'r--', 'LineWidth', 1 ,'DisplayName','PDF')
xlabel('Samples')
ylabel('mV')
legend



