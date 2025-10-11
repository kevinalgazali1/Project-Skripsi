<?php

namespace App\Exports;

use App\Models\Loker;
use Illuminate\Support\Facades\Auth;
use Maatwebsite\Excel\Concerns\WithHeadings;
use Maatwebsite\Excel\Concerns\FromCollection;

class LokerExport implements FromCollection, WithHeadings
{
    protected $request;

    public function __construct($request)
    {
        $this->request = $request;
    }

    public function collection()
    {
        $user = Auth::user();

        $query = Loker::with(['user.mitraProfile', 'daftarLoker'])
            ->withCount('daftarLoker')
            ->orderBy('created_at', 'desc');

        // Jika bukan admin, filter berdasarkan user_id
        if ($user->role !== 'admin') {
            $query->where('user_id', $user->id);
        }

        // Terapkan semua filter
        if ($this->request->filled('nama_perusahaan')) {
            $query->where('nama_perusahaan', 'like', '%' . $this->request->nama_perusahaan . '%');
        }

        if ($this->request->filled('lokasi')) {
            $query->where('lokasi', 'like', '%' . $this->request->lokasi . '%');
        }

        if ($this->request->filled('tipe')) {
            $query->where('tipe', $this->request->tipe);
        }

        if ($this->request->filled('status')) {
            $query->where('status', $this->request->status);
        }

        $loker = $query->get();

        return $loker->map(function ($p) {
            $pesertas = collect($p->daftarLoker);

            return [
                'Instansi' => $p->user->mitraProfile->nama_instansi ?? '-',
                'Nama Pelatihan' => $p->nama_perusahaan ?? '-',
                'Lokasi' => $p->lokasi ?? '-',
                'Posisi' => $p->posisi ?? '-',
                'Jumlah Peserta' => $p->daftar_loker_count ?? 0,
                'Jumlah Asal Sekolah' => $pesertas->pluck('asal_sekolah')->unique()->count() ?? 0,
                'Jumlah Jurusan' => $pesertas->pluck('jurusan')->unique()->count() ?? 0,
                'Jumlah Bekerja/Wirausaha' => $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count() ?? 0,
                'Jumlah Tidak Bekerja' => $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count() ?? 0,
                'Jumlah Kuliah' => $pesertas->where('status_saat_ini', 'Kuliah')->count() ?? 0,
            ];
        });
    }

    public function headings(): array
    {
        return [
            'Instansi',
            'Nama Pelatihan',
            'Lokasi',
            'Posisi',
            'Jumlah Peserta',
            'Jumlah Asal Sekolah',
            'Jumlah Jurusan',
            'Jumlah Bekerja/Wirausaha',
            'Jumlah Tidak Bekerja',
            'Jumlah Kuliah',
        ];
    }
}
