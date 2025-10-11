<?php

namespace App\Exports;

use App\Models\DaftarLoker;
use Maatwebsite\Excel\Concerns\WithHeadings;
use Maatwebsite\Excel\Concerns\FromCollection;

class PesertaLokerExport implements FromCollection, WithHeadings
{
    protected $loker_id;

    public function __construct($loker_id)
    {
        $this->loker_id = $loker_id;
    }

    public function collection()
    {
        $data = DaftarLoker::where('loker_id', $this->loker_id)->get();

        return $data->map(function ($item) {
            return [
                $item->nama_lengkap,
                $item->email,
                $item->no_hp,
                $item->asal_sekolah,
                $item->jurusan,
                $item->jenis_kelamin,
                $item->tanggal_lahir,
                "'" . $item->nik,
                $item->tahun_kelulusan,
                $item->npsn,
                $item->provinsi,
                $item->kota,
                $item->alamat,
                $item->status_saat_ini,
                $item->bidang_pekerjaan,
                $item->sertifikasi_terakhir,
                $item->kesesuaian_sertifikasi,
                $item->nama_universitas,
                $item->jurusan_universitas,
            ];
        });
    }

    public function headings(): array
    {
        return [
            'Nama Lengkap',
            'Email',
            'No. Telepon',
            'Asal Sekolah',
            'Jurusan',
            'Jenis Kelamin',
            'Tanggal Lahir',
            'NIK',
            'Tahun Kelulusan',
            'NPSN',
            'Provinsi',
            'Kota',
            'Alamat',
            'Status Saat Ini',
            'Bidang Pekerjaan',
            'Sertifikasi Terakhir',
            'Kesesuaian Sertifikasi',
            'Nama Universitas',
            'Jurusan Universitas',
        ];
    }
}

