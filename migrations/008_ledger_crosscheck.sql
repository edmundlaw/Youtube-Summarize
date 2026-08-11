-- A second, independent reading of each figure, from our own ASR.
--
-- The validator compares a summary against a ledger built from the same
-- transcript, so a mis-heard number is mis-heard identically on both sides and
-- passes. Measured on MgN00MCDDRM: the captions say 中芯 北水淨流入 29億 and
-- contradict themselves thirty seconds later with 接近三百億; two ASR models
-- independently hear 299億. A 10x error, invisible to every existing check.
--
-- NULL means never checked, which is what every existing row is. It must never
-- be read as agreement.
ALTER TABLE number_ledger ADD COLUMN asr_normalized TEXT;
ALTER TABLE number_ledger ADD COLUMN crosscheck TEXT;
ALTER TABLE number_ledger ADD COLUMN asr_model TEXT;
