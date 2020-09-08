#!/usr/bin/perl
$outFile = shift; 

%results = ();
$testFile = shift;
$predFile = shift; 
@predLines = ();
open(FI, "< $predFile") or die "Open $predFile failed.";
while(<FI>){
	chomp($_);
	push(@predLines,$_);
}#FI
close(FI);

open(FI, "< $testFile") or die "Open $testFile failed.";
$i = 0;
while(<FI>){
	chomp($_);
	@tmp = split(' ', $_);
	@currQID = split(/:/, $tmp[1]);
	$qID = $currQID[1];
	$dID = $tmp[$#tmp];
	$results{$qID}{$dID} = $predLines[$i];
	$i++;
}#FI
close(FI);

# output
open(FO, "> $outFile") or die "Open $outFile failed.";
foreach $qID (sort keys %results){
	chomp($qID);
	$i = 1;
	foreach $dID (sort {$results{$qID}{$b}<=>$results{$qID}{$a} || $a cmp $b} 
	keys %{ $results{$qID} }){
		print FO "$qID Q0 $dID $i $results{$qID}{$dID} indri\n";
		$i ++;
	}
}
close(FO);
