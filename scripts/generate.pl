#!/usr/bin/perl
use POSIX qw/floor/;
$featuresDir = shift;
$workingSetFile = shift;

@streamList = ("doc");
@featureList1 = ("CoverQueryNum","CoverQueryRatio","Len","IDF","BM25","LMIR.DIR","LMIR.JM","Ent","StopCover","FracStops");
@featureList2 = ("TF","TFNorm","TFIDF");
@aggregators = ("Sum","Min","Max","Mean","Var");
$aggNum = $#aggregators + 1;

%initDocs = ();
open(FI, "< $workingSetFile") or die "Open $workingSetFile failed.\n";
while(<FI>){
	($qID,$Q0,$dID,$pos,$score,$indri) = split(/ / ,$_);
	$initDocs{$qID}{$dID} = $pos;
}#while
close(FI);

# get features values
%features = ();
%featureID = ();
%noStream = ();
$fID = 0;
$outFile = "featureID";
open(FO, "> $outFile") or die "Open $outFile failed.";
foreach $feature (@featureList1){
	foreach $stream (@streamList){
		$featureName = "${stream}${feature}";
		$fID ++;
		$featureID{$featureName} = $fID;
		print FO "$featureName $fID\n";
		foreach $qID (keys %initDocs){
			$featureFile = "${featuresDir}/${stream}${feature}_${qID}";
			if (-e "$featureFile"){
				open(FI, "< $featureFile") or die "Open $featureFile failed.";
				while(<FI>){
					chomp($_);
					($dID,$score) = split(/ / ,$_);
					if (exists $initDocs{$qID}{$dID}){
						$features{$featureName}{$qID}{$dID} = $score;
					}#if
				}#while
				close(FI);
			}#if
			else{
				$noStream{$featureName}{$qID} = 1;
			}#else
		}#qID
	}#stream
}#feature

foreach $feature (@featureList2){
foreach $stream (@streamList){
	for ($agg = 0; $agg < $aggNum; $agg ++){
		$featureName = "${stream}$aggregators[$agg]${feature}";
		$fID ++;
		$featureID{$featureName} = $fID;
		print FO "$featureName $fID\n";
	}#agg
	foreach $qID (keys %initDocs){
		$featureFile = "${featuresDir}/${stream}${feature}_${qID}";
		if (-e "$featureFile"){
			open(FI, "< $featureFile") or die "Open $featureFile failed.";
			while(<FI>){
				chomp($_);
				($dID,@scores) = split(/ / ,$_);
				if (exists $initDocs{$qID}{$dID}){
					for ($agg = 0; $agg < $aggNum; $agg ++){
						$featureName = "${stream}$aggregators[$agg]${feature}";
						$features{$featureName}{$qID}{$dID} = $scores[$agg];
					}#agg
				}#if
			}#FI
			close(FI);
		}
		else{
			for ($agg = 0; $agg < $aggNum; $agg ++){
				$featureName = "${stream}$aggregators[$agg]${feature}";
				$noStream{$featureName}{$qID} = 1;
			}#agg
		}#else
	}#qID
}#stream
}#feature
close(FO);

# get features minimum and maximum
%min = ();
%max = ();
foreach $featureName (keys %features){
foreach $qID (keys %initDocs){
if(exists $noStream{$featureName}{$qID}){
	$min{$featureName}{$qID} = 0;
	$max{$featureName}{$qID} = 0;
}
else{
	foreach $dID (keys %{ $features{$featureName}{$qID} }){
		if(!exists $min{$featureName}{$qID} ||
			$min{$featureName}{$qID} > $features{$featureName}{$qID}{$dID}){
			$min{$featureName}{$qID} = $features{$featureName}{$qID}{$dID};
		}
		if(!exists $max{$featureName}{$qID} || 
			$max{$featureName}{$qID} < $features{$featureName}{$qID}{$dID}){
			$max{$featureName}{$qID} = $features{$featureName}{$qID}{$dID};
		}
	}#dID
}#else
}#qID
# if no documents were found with features
foreach $qID (keys %initDocs){
	if(!exists $min{$featureName}{$qID}){
		$min{$featureName}{$qID} = 0;
	}
	if(!exists $max{$featureName}{$qID}){
		$max{$featureName}{$qID} = 0;
	}
}#qID
}#featureName

# print
$outFile = "features";
open(FO, "> $outFile") or die "Open $outFile failed.";
foreach $qID (sort {$a<=>$b} keys %initDocs){
	foreach $dID (sort {$initDocs{$qID}{$a}<=>$initDocs{$qID}{$b}} keys %{ $initDocs{$qID} }){
	print FO "0 qid:$qID";
	foreach $featureName (sort {$featureID{$a}<=>$featureID{$b}} keys %featureID){
		$res = 0;
		if (exists $features{$featureName}{$qID}{$dID} && 
			$min{$featureName}{$qID} < $max{$featureName}{$qID}){
			$top = $features{$featureName}{$qID}{$dID} - $min{$featureName}{$qID};
			$bottom = $max{$featureName}{$qID} - $min{$featureName}{$qID};
			$res = $top / $bottom;
		}
		$res = sprintf("%.8f",$res);
		print FO " $featureID{$featureName}:${res}";	
	}#featureName
	print FO " \# $dID\n";
	}#dID
}#qID
close(FO);
