#!/usr/bin/perl

#opendir my $dh, "./local/" or die "failed";
my @files = glob("/home/lei/dataset/voxceleb2/vox/wav/id10001/*/*.wav");
foreach (@files){
#    print $_ . "\n";
    $wav = $_;
    $rec_id = substr($wav, -21, 11);
    $segment = substr($wav, -9, 5);
    print "$rec_id $segment \n"
}
