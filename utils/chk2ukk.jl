#!/usr/bin/env -S julia --project=/home/jqiao/git/aiida-qe-super/utils
# Convert W90 chk U matrices to the EPW ukk format.
# Usage:
#   julia chk2ukk.jl graphene.chk qe_out.xml graphene.ukk
using Dates
using WannierIO

if length(ARGS) != 3
    error("Only accept three arguments")
end
chk_filename = ARGS[1]
qexml_filename = ARGS[2]
ukk_filename = ARGS[3]

chk = read_chk(chk_filename)
alat = WannierIO.read_qe_xml(qexml_filename).alat
ukk = WannierIO.Ukk(chk, alat)
WannierIO.write_epw_ukk(ukk_filename, ukk)

println("Job done at ", Dates.now())
